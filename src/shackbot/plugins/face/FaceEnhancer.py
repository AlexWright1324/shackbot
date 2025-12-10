import cv2
import onnxruntime
import numpy as np

from .models import WARP_TEMPLATES, FaceEnhancerModel

from .Face import Face


class FaceEnhancer:
    def __init__(self, model: FaceEnhancerModel):
        self.model = model
        self.session = onnxruntime.InferenceSession(model["path"])

    def create_static_box_mask(
        self,
        crop_size: cv2.typing.Size,
        face_mask_blur: float,
        face_mask_padding: tuple[int, int, int, int],
    ) -> np.ndarray:
        blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
        blur_area = max(blur_amount // 2, 1)
        box_mask = np.ones(crop_size, np.float32)
        box_mask[
            : max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :
        ] = 0
        box_mask[
            -max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)) :, :
        ] = 0
        box_mask[
            :, : max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))
        ] = 0
        box_mask[
            :, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)) :
        ] = 0
        if blur_amount > 0:
            box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
        return box_mask

    def prepare_crop_frame(self, crop_vision_frame: np.ndarray) -> np.ndarray:
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
        crop_vision_frame = np.expand_dims(
            crop_vision_frame.transpose(2, 0, 1), axis=0
        ).astype(np.float32)
        return crop_vision_frame

    def estimate_matrix_by_face_landmark_5(
        self,
        face_landmark_5: np.ndarray,
        warp_template: str,
        crop_size: cv2.typing.Size,
    ) -> np.ndarray:
        template = WARP_TEMPLATES.get(warp_template)
        if template is None:
            raise ValueError(f"Warp template {warp_template} not found")
        normed_warp_template = template * crop_size
        affine_matrix = cv2.estimateAffinePartial2D(
            face_landmark_5,
            normed_warp_template,
            method=cv2.RANSAC,
            ransacReprojThreshold=100,
        )[0]
        return affine_matrix

    def warp_face_by_face_landmark_5(
        self,
        temp_vision_frame: np.ndarray,
        face_landmark_5: np.ndarray,
        warp_template: str,  # the ones defined in model templates
        crop_size: cv2.typing.Size,
    ) -> tuple[np.ndarray, np.ndarray]:
        affine_matrix = self.estimate_matrix_by_face_landmark_5(
            face_landmark_5, warp_template, crop_size
        )
        crop_vision_frame = cv2.warpAffine(
            temp_vision_frame,
            affine_matrix,
            crop_size,
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_AREA,
        )
        return crop_vision_frame, affine_matrix

    def normalize_crop_frame(self, crop_vision_frame: np.ndarray) -> np.ndarray:
        crop_vision_frame = np.clip(crop_vision_frame, -1, 1)
        crop_vision_frame = (crop_vision_frame + 1) / 2
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        crop_vision_frame = (crop_vision_frame * 255.0).round()
        crop_vision_frame = crop_vision_frame.astype(np.uint8)[:, :, ::-1]
        return crop_vision_frame

    def apply_enhance(self, crop_vision_frame: np.ndarray):
        frame_processor_inputs = {}
        for frame_processor_input in self.session.get_inputs():
            if frame_processor_input.name == "input":
                frame_processor_inputs[frame_processor_input.name] = crop_vision_frame
            if frame_processor_input.name == "weight":
                weight = np.ndarray([1]).astype(np.double)
                frame_processor_inputs[frame_processor_input.name] = weight

        result = self.session.run(None, frame_processor_inputs)
        crop_vision_frame = np.array(result[0])[0]
        return crop_vision_frame

    def paste_back(
        self,
        temp_vision_frame: np.ndarray,
        crop_vision_frame: np.ndarray,
        crop_mask: np.ndarray,
        affine_matrix: np.ndarray,
    ):
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        temp_size = temp_vision_frame.shape[:2][::-1]
        inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
        inverse_vision_frame = cv2.warpAffine(
            crop_vision_frame,
            inverse_matrix,
            temp_size,
            borderMode=cv2.BORDER_REPLICATE,
        )
        paste_vision_frame = temp_vision_frame.copy()
        paste_vision_frame[:, :, 0] = (
            inverse_mask * inverse_vision_frame[:, :, 0]
            + (1 - inverse_mask) * temp_vision_frame[:, :, 0]
        )
        paste_vision_frame[:, :, 1] = (
            inverse_mask * inverse_vision_frame[:, :, 1]
            + (1 - inverse_mask) * temp_vision_frame[:, :, 1]
        )
        paste_vision_frame[:, :, 2] = (
            inverse_mask * inverse_vision_frame[:, :, 2]
            + (1 - inverse_mask) * temp_vision_frame[:, :, 2]
        )
        return paste_vision_frame

    def blend_frame(
        self,
        temp_vision_frame: np.ndarray,
        paste_vision_frame: np.ndarray,
        blend_strength: float = 0.5,  # value between 0 and 1
    ) -> np.ndarray:
        face_enhancer_blend = 1 - blend_strength
        temp_vision_frame = cv2.addWeighted(
            temp_vision_frame,
            face_enhancer_blend,
            paste_vision_frame,
            1 - face_enhancer_blend,
            0,
        )
        return temp_vision_frame

    def enhance_face(
        self, target_face: Face, temp_vision_frame: np.ndarray
    ) -> np.ndarray:
        landmark = target_face.kps  # get('landmark_2d_106')
        if landmark is None:
            raise ValueError("Face landmark data is required for enhancement")

        crop_vision_frame, affine_matrix = self.warp_face_by_face_landmark_5(
            temp_vision_frame, landmark, self.model["template"], self.model["size"]
        )
        box_mask = self.create_static_box_mask(
            crop_vision_frame.shape[:2][::-1],
            face_mask_blur=0.0,  # (0, 0, 0, 0),  # facefusion.globals.face_mask_blur,
            face_mask_padding=(0, 0, 0, 0),
        )
        crop_mask_list = [box_mask]

        # if 'occlusion' in facefusion.globals.face_mask_types:
        #    occlusion_mask = create_occlusion_mask(crop_vision_frame)
        #    crop_mask_list.append(occlusion_mask)

        crop_vision_frame = self.prepare_crop_frame(crop_vision_frame)
        crop_vision_frame = self.apply_enhance(crop_vision_frame=crop_vision_frame)
        crop_vision_frame = self.normalize_crop_frame(crop_vision_frame)
        crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)
        paste_vision_frame = self.paste_back(
            temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix
        )
        temp_vision_frame = self.blend_frame(temp_vision_frame, paste_vision_frame)
        return temp_vision_frame
