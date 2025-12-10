from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm as l2norm


@dataclass
class Face:
    """
    Represents a detected face with its associated features and embeddings.

    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2] in pixels
        det_score: Detection confidence score (0.0 to 1.0)
        kps: Facial keypoints, typically 5 points (2 eyes, nose, 2 mouth corners)
        landmark_3d_68: 3D facial landmarks (68 points)
        landmark_2d_106: 2D facial landmarks (106 points)
        embedding: Face embedding vector for recognition
        gender: Gender prediction (0: female, 1: male, None: not computed)
        age: Predicted age in years
    """

    # Detection attributes (always present)
    bbox: NDArray[np.float32]  # Shape: (4,) - [x1, y1, x2, y2]
    det_score: float

    # Keypoints (5 facial landmarks for alignment)
    kps: Optional[NDArray[np.float32]] = None  # Shape: (5, 2)

    # Detailed landmarks
    landmark_3d_68: Optional[NDArray[np.float32]] = None  # Shape: (68, 3)
    landmark_2d_106: Optional[NDArray[np.float32]] = None  # Shape: (106, 2)

    # Face recognition embedding
    embedding: Optional[NDArray[np.float32]] = None  # Shape: (512,) typically

    # Demographic predictions
    gender: Optional[int] = None  # 0: female, 1: male
    age: Optional[int] = None

    # Pose estimation (rotation angles)
    pose: Optional[NDArray[np.float32]] = None  # Shape: (3,) - [pitch, yaw, roll]

    @property
    def embedding_norm(self) -> Optional[float]:
        """Compute L2 norm of the face embedding."""
        if self.embedding is None:
            return None
        return float(l2norm(self.embedding))

    @property
    def normed_embedding(self) -> Optional[NDArray[np.float32]]:
        """Get normalized face embedding (unit vector)."""
        if self.embedding is None:
            return None
        norm = self.embedding_norm
        if norm is None or norm == 0:
            return None
        return self.embedding / norm

    @property
    def sex(self) -> Optional[str]:
        """Get gender as a string ('M' or 'F')."""
        if self.gender is None:
            return None
        return "M" if self.gender == 1 else "F"

    @property
    def bbox_area(self) -> float:
        """Calculate the area of the bounding box in pixels."""
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        return float(width * height)

    @property
    def bbox_center(self) -> tuple[float, float]:
        """Get the center point of the bounding box (x, y)."""
        x_center = (self.bbox[0] + self.bbox[2]) / 2
        y_center = (self.bbox[1] + self.bbox[3]) / 2
        return (float(x_center), float(y_center))

    @property
    def bbox_width(self) -> float:
        """Get the width of the bounding box in pixels."""
        return float(self.bbox[2] - self.bbox[0])

    @property
    def bbox_height(self) -> float:
        """Get the height of the bounding box in pixels."""
        return float(self.bbox[3] - self.bbox[1])

    def has_embedding(self) -> bool:
        """Check if face has a valid embedding."""
        return self.embedding is not None and self.embedding.size > 0

    def has_landmarks(self) -> bool:
        """Check if face has any landmark data."""
        return (
            self.kps is not None
            or self.landmark_3d_68 is not None
            or self.landmark_2d_106 is not None
        )

    def compute_similarity(self, other: "Face") -> Optional[float]:
        """
        Compute cosine similarity between this face and another face.

        Args:
            other: Another Face object to compare with

        Returns:
            Similarity score between -1 and 1, or None if embeddings are missing
        """
        if not self.has_embedding() or not other.has_embedding():
            return None

        # Use normalized embeddings for cosine similarity
        embedding1 = self.normed_embedding
        embedding2 = other.normed_embedding

        if embedding1 is None or embedding2 is None:
            return None

        similarity = float(np.dot(embedding1, embedding2))
        return similarity

    def __repr__(self) -> str:
        """Provide a clean string representation of the Face."""
        parts = [
            f"Face(bbox=[{self.bbox[0]:.1f}, {self.bbox[1]:.1f}, {self.bbox[2]:.1f}, {self.bbox[3]:.1f}]",
            f"score={self.det_score:.3f}",
        ]

        if self.has_embedding():
            parts.append("embedding=✓")
        if self.kps is not None:
            parts.append("kps=✓")
        if self.gender is not None:
            parts.append(f"sex={self.sex}")
        if self.age is not None:
            parts.append(f"age={self.age}")

        return ", ".join(parts) + ")"
