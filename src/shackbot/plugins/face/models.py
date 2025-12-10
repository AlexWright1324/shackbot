import os
from typing import NotRequired, TypedDict
import numpy as np

MODELS_DIR = os.getenv("MODELS_DIR", "models")


class Model(TypedDict):
    url: str
    path: str
    extract: NotRequired[bool]


class FaceEnhancerModel(Model):
    template: str
    size: tuple[int, int]


SWAPPER: Model = {
    "url": "https://socaityfiles.blob.core.windows.net/model-zoo/face2face/insightface/inswapper_128.onnx",
    "path": f"{MODELS_DIR}/inswapper_128.onnx",
}

INSIGHT: Model = {
    "url": "https://socaityfiles.blob.core.windows.net/model-zoo/face2face/insightface/buffalo_l.zip",
    "path": f"{MODELS_DIR}/buffalo_l",
    "extract": True,
}

FACE_ENHANCER_MODELS: dict[str, FaceEnhancerModel] = {
    "gfpgan_1.4": {
        "url": "https://socaityfiles.blob.core.windows.net/model-zoo/face2face/face_enhancer/gfpgan_1.4.onnx",
        "path": f"{MODELS_DIR}/face_enhancer/gfpgan_1.4.onnx",
        "template": "ffhq_512",
        "size": (512, 512),
    },
    "gpen_bfr_256": {
        "url": "https://socaityfiles.blob.core.windows.net/model-zoo/face2face/face_enhancer/gpen_bfr_256.onnx",
        "path": f"{MODELS_DIR}/face_enhancer/gpen_bfr_256.onnx",
        "template": "arcface_128_v2",
        "size": (256, 256),
    },
    "gpen_bfr_512": {
        "url": "https://socaityfiles.blob.core.windows.net/model-zoo/face2face/face_enhancer/gpen_bfr_512.onnx",
        "path": f"{MODELS_DIR}/face_enhancer/gpen_bfr_512.onnx",
        "template": "ffhq_512",
        "size": (512, 512),
    },
    "gpen_bfr_1024": {
        "url": "https://socaityfiles.blob.core.windows.net/model-zoo/face2face/face_enhancer/gpen_bfr_1024.onnx",
        "path": f"{MODELS_DIR}/face_enhancer/gpen_bfr_1024.onnx",
        "template": "ffhq_512",
        "size": (1024, 1024),
    },
    "gpen_bfr_2048": {
        "url": "https://socaityfiles.blob.core.windows.net/model-zoo/face2face/face_enhancer/gpen_bfr_2048.onnx",
        "path": f"{MODELS_DIR}/face_enhancer/gpen_bfr_2048.onnx",
        "template": "ffhq_512",
        "size": (2048, 2048),
    },
}

WARP_TEMPLATES = {
    "arcface_112_v1": np.array(
        [
            [0.35473214, 0.45658929],
            [0.64526786, 0.45658929],
            [0.50000000, 0.61154464],
            [0.37913393, 0.77687500],
            [0.62086607, 0.77687500],
        ]
    ),
    "arcface_112_v2": np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089],
        ]
    ),
    "arcface_128_v2": np.array(
        [
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453],
        ]
    ),
    "ffhq_512": np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ]
    ),
}
