from __future__ import division
from os import PathLike
import numpy as np
from numpy.typing import NDArray
import cv2
import onnx
import onnxruntime
from .math import norm_crop

from .Face import Face


class ArcFace:
    def __init__(self, model_file: str | PathLike[str]):
        self.taskname = "recognition"
        find_sub = False
        find_mul = False
        model = onnx.load(model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            # print(nid, node.name)
            if node.name.startswith("Sub") or node.name.startswith("_minus"):
                find_sub = True
            if node.name.startswith("Mul") or node.name.startswith("_mul"):
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
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
        self.output_shape = outputs[0].shape

    def get(self, img: cv2.typing.MatLike, face: "Face"):
        """
        Extract face embedding from the detected face region.

        Args:
            img: Input image in BGR format
            face: Face object with kps (keypoints) attribute set

        Returns:
            Face embedding vector, also stored in face.embedding
        """
        if face.kps is None:
            raise ValueError(
                "Face keypoints (kps) are required for embedding extraction"
            )

        # Align and crop face using keypoints
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])

        # Extract embedding
        embedding = self.get_feat(aimg).flatten().astype(np.float32)
        face.embedding = embedding

        return embedding

    def compute_sim(
        self, feat1: cv2.typing.MatLike, feat2: cv2.typing.MatLike
    ) -> float:
        """
        Compute cosine similarity between two face embeddings.

        Args:
            feat1: First face embedding
            feat2: Second face embedding

        Returns:
            Similarity score between -1 and 1
        """
        from numpy.linalg import norm

        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return float(sim)

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(
            imgs,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out
