# -*- coding: utf-8 -*-
import numpy as np
from numpy.typing import NDArray
import onnxruntime
import cv2
import onnx
from onnx import numpy_helper
from .math import norm_crop2


from .Face import Face


class FaceSwapper:
    INPUT_MEAN = 0.0
    INPUT_STD = 255.0

    def __init__(self, model_file: str):
        with open(model_file, "rb") as f:
            model_bytes = f.read()

        # FIXME: This ends up parsing the model twice
        model = onnx.load_model_from_string(model_bytes)
        self.emap = numpy_helper.to_array(model.graph.initializer[-1])
        self.session = onnxruntime.InferenceSession(model_bytes)

        inputs = self.session.get_inputs()
        self.input_size = tuple(inputs[0].shape[2:4][::-1])
        self.input_names = [inp.name for inp in inputs]

        outputs = self.session.get_outputs()
        self.output_names = [out.name for out in outputs]

        assert len(self.output_names) == 1

    def forward(self, img: cv2.typing.MatLike, latent):
        img = (img - self.INPUT_MEAN) / self.INPUT_STD
        pred = self.session.run(
            self.output_names, {self.input_names[0]: img, self.input_names[1]: latent}
        )
        return pred[0]

    def swap(
        self,
        img: cv2.typing.MatLike,
        target_face: Face,
        source_face: Face,
        paste_back: bool = True,
    ):
        """
        Swap the target face with the source face in the image.

        Args:
            img: Input image in BGR format
            target_face: The face in the image to be replaced
            source_face: The face to swap in (must have embedding)
            paste_back: If True, blend result back into original image; if False, return crop

        Returns:
            If paste_back=True: The full image with swapped face
            If paste_back=False: Tuple of (swapped face crop, affine transform matrix)
        """
        if target_face.kps is None:
            raise ValueError("Target face must have keypoints (kps)")
        if source_face.normed_embedding is None:
            raise ValueError("Source face must have a valid embedding")
        aimg, M = norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.INPUT_STD,
            self.input_size,
            (self.INPUT_MEAN, self.INPUT_MEAN, self.INPUT_MEAN),
            swapRB=True,
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(
            self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent}
        )[0]

        # print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return bgr_fake, M
        else:
            target_img = img
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2, :] = 0
            fake_diff[-2:, :] = 0
            fake_diff[:, :2] = 0
            fake_diff[:, -2:] = 0
            IM = cv2.invertAffineTransform(M)
            img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
            bgr_fake = cv2.warpAffine(
                bgr_fake,
                IM,
                (target_img.shape[1], target_img.shape[0]),
                borderValue=0.0,
            )
            img_white = cv2.warpAffine(
                img_white,
                IM,
                (target_img.shape[1], target_img.shape[0]),
                borderValue=0.0,
            )
            fake_diff = cv2.warpAffine(
                fake_diff,
                IM,
                (target_img.shape[1], target_img.shape[0]),
                borderValue=0.0,
            )
            img_white[img_white > 20] = 255
            fthresh = 10
            fake_diff[fake_diff < fthresh] = 0
            fake_diff[fake_diff >= fthresh] = 255
            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask == 255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h * mask_w))
            k = max(mask_size // 10, 10)
            # k = max(mask_size//20, 6)
            # k = 6
            kernel = np.ones((k, k), np.uint8)
            img_mask = cv2.erode(img_mask, kernel, iterations=1)
            kernel = np.ones((2, 2), np.uint8)
            fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
            k = max(mask_size // 20, 5)
            # k = 3
            # k = 3
            kernel_size = (k, k)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            k = 5
            kernel_size = (k, k)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
            img_mask /= 255
            fake_diff /= 255
            # img_mask = fake_diff
            img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
            fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(
                np.float32
            )
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged
