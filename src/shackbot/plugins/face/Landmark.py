import onnx
import onnxruntime
import cv2
import numpy as np
from numpy.typing import NDArray

from os import path, PathLike
import pickle

from .math import (
    trans_points,
    transform,
    estimate_affine_matrix_3d23d,
    P2sRt,
    matrix2angle,
)

from .Face import Face


class Landmark:
    def __init__(self, model_file: str | PathLike[str]):
        self.model_file = model_file
        find_sub = False
        find_mul = False
        model = onnx.load(model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            if node.name.startswith("Sub") or node.name.startswith("_minus"):
                find_sub = True
            if node.name.startswith("Mul") or node.name.startswith("_mul"):
                find_mul = True
            if nid < 3 and node.name == "bn_data":
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 128.0

        self.input_mean = input_mean
        self.input_std = input_std

        self.session = onnxruntime.InferenceSession(model_file)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        output_shape = outputs[0].shape
        self.require_pose = False
        self.mean_lmk = None
        # print('init output_shape:', output_shape)
        if output_shape[1] == 3309:
            self.lmk_dim = 3
            self.lmk_num = 68
            self.mean_lmk = self._get_meanshape_68()
            self.require_pose = True
        else:
            self.lmk_dim = 2
            self.lmk_num = output_shape[1] // self.lmk_dim
        self.taskname = "landmark_%dd_%d" % (self.lmk_dim, self.lmk_num)

    def _get_meanshape_68(self):
        if self.mean_lmk is not None:
            return self.mean_lmk

        meanshape_68_path = path.join(path.dirname(self.model_file), "meanshape_68.pkl")
        with open(meanshape_68_path, "rb") as f:
            obj = pickle.load(f)

        self.mean_lmk = obj
        return self.mean_lmk

    def get(self, img: cv2.typing.MatLike, face: Face):
        """
        Extract facial landmarks from the detected face region.

        Args:
            img: Input image in BGR format
            face: Face object with bbox attribute set

        Returns:
            Landmark coordinates, also stored in face.landmark_3d_68 or face.landmark_2d_106
        """
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h) * 1.5)

        # Transform image to model input size
        aimg, M = transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )

        # Run model inference
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0]

        # Reshape predictions based on dimensionality
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))

        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1 :, :]

        # Scale landmarks back to input image coordinates
        pred[:, 0:2] += 1
        pred[:, 0:2] *= self.input_size[0] // 2
        if pred.shape[1] == 3:
            pred[:, 2] *= self.input_size[0] // 2

        # Transform landmarks back to original image space
        IM = cv2.invertAffineTransform(M)
        pred = trans_points(pred, IM)

        # Store landmarks in the appropriate face attribute
        pred_typed = pred.astype(np.float32)
        if self.lmk_dim == 3 and self.lmk_num == 68:
            face.landmark_3d_68 = pred_typed
        elif self.lmk_dim == 2 and self.lmk_num == 106:
            face.landmark_2d_106 = pred_typed

        # Estimate head pose if required
        if self.require_pose:
            P = estimate_affine_matrix_3d23d(self.mean_lmk, pred)
            s, R, t = P2sRt(P)
            rx, ry, rz = matrix2angle(R)
            face.pose = np.array([rx, ry, rz], dtype=np.float32)

        return pred_typed
