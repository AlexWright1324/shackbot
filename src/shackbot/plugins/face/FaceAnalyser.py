"""
Face analysis module for detecting and analyzing faces in images.

This module provides the Face dataclass and FaceAnalyser class which orchestrates
face detection, landmark extraction, and face recognition using multiple specialized models.
"""

import os
import cv2
import numpy as np

from .RetinaFace import RetinaFace
from .Landmark import Landmark
from .ArcFace import ArcFace
from .Face import Face


class FaceAnalyser:
    """
    High-level face analysis orchestrator.

    Coordinates face detection, landmark extraction, and embedding generation
    using specialized models. Provides a simple interface for analyzing faces
    in images.

    Attributes:
        retina_face: Face detection model
        landmark: 3D landmark detection model (68 points)
        landmark2d: 2D landmark detection model (106 points)
        arcface: Face recognition embedding model
    """

    def __init__(self, models_path: str):
        """
        Initialize face analyzer with models from the given path.

        Args:
            models_path: Directory containing ONNX model files
        """
        # Load models
        self.retina_face = RetinaFace(os.path.join(models_path, "det_10g.onnx"))
        self.landmark = Landmark(os.path.join(models_path, "1k3d68.onnx"))
        self.landmark2d = Landmark(os.path.join(models_path, "2d106det.onnx"))
        self.arcface = ArcFace(os.path.join(models_path, "w600k_r50.onnx"))

    def analyze(
        self,
        img: cv2.typing.MatLike,
        max_num: int = 0,
        extract_embedding: bool = True,
        extract_landmarks: bool = True,
    ) -> list[Face]:
        """
        Detect and analyze all faces in an image.

        Args:
            img: Input image in BGR format (OpenCV format)
            max_num: Maximum number of faces to detect (0 = unlimited)
            extract_embedding: Whether to compute face embeddings
            extract_landmarks: Whether to extract facial landmarks

        Returns:
            List of Face objects with detected features
        """
        # Detect faces
        bboxes, kpss = self.retina_face.detect(img, max_num=max_num, metric="default")
        if bboxes.shape[0] == 0:
            return []

        faces: list[Face] = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4].astype(np.float32)
            det_score = float(bboxes[i, 4])
            kps = None if kpss is None else kpss[i].astype(np.float32)

            # Create Face object with detection results
            face = Face(bbox=bbox, kps=kps, det_score=det_score)

            # Extract landmarks if requested
            if extract_landmarks:
                self.landmark.get(img, face)
                self.landmark2d.get(img, face)

            # Extract embedding if requested
            if extract_embedding and face.kps is not None:
                self.arcface.get(img, face)

            faces.append(face)

        return faces
