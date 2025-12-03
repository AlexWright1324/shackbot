# Face2Face Library (Local Copy)

This is a local copy of the `socaity-face2face` library, integrated into the shackbot project for easier customization and extension.

## Structure

```
face2face/
├── __init__.py                 # Main entry point, exports Face2Face class
├── settings.py                 # Configuration settings (model paths, device settings)
├── model_definitions.py        # Model download URLs and configurations
├── server.py                   # (Optional) Server functionality
├── core/
│   ├── face2face.py           # Main Face2Face class with all functionality
│   ├── compatibility/         # InsightFace-compatible implementations
│   │   ├── Face.py           # Face data structure
│   │   ├── FaceAnalysis.py   # Face detection and analysis
│   │   ├── INSwapper.py      # Face swapping model
│   │   ├── ArcFaceONNX.py    # Face recognition model
│   │   ├── Landmark.py       # Face landmark detection
│   │   ├── Attribute.py      # Face attribute detection (age, gender)
│   │   ├── retinaface.py     # Face detection backend
│   │   ├── face_align.py     # Face alignment utilities
│   │   └── transform.py      # Geometric transformations
│   ├── mixins/                # Feature mixins
│   │   ├── _face_embedding.py    # Face embedding management
│   │   ├── _face_enhance.py      # Face enhancement (GFPGAN, GPEN)
│   │   ├── _face_recognition.py  # Face recognition and matching
│   │   ├── _image_swap.py        # Image face swapping
│   │   └── _video_swap.py        # Video face swapping
│   └── modules/               # Utility modules
│       ├── utils.py           # Helper functions (download, image loading)
│       ├── file_writable_face.py  # Serializable Face objects
│       └── face_enhance/      # Face enhancement models
│           ├── face_enhancer.py      # Enhancement pipeline
│           ├── face_enhance_models.py # Model configurations
│           └── face_occlusion.py     # Occlusion detection
```

## Main Features

### 1. Face Swapping
- **swap_img_to_img**: Swap faces between two images
- **swap_to_faces**: Swap faces to registered face embeddings
- **swap_video**: Swap faces in videos
- **swap_pairs**: Recognition-based face swapping (swap specific people)

### 2. Face Enhancement
- **enhance_faces**: Enhance all faces in an image
- **enhance_single_face**: Enhance a specific face
- Models: GFPGAN, GPEN (256, 512, 1024, 2048)

### 3. Face Recognition
- **face_recognition**: Recognize faces against a database
- **calculate_face_distances**: Calculate similarity between faces
- **detect_faces**: Detect faces in images

### 4. Face Embeddings
- **add_face**: Add face embeddings to the database
- **get_faces**: Load face embeddings from various sources
- **load_all_face_embeddings**: Load all saved embeddings

## Usage in Shackbot

Currently used in `plugins/kirkify.py`:
```python
from face2face import Face2Face

# Initialize
f2f = Face2Face()

# Add a reference face
f2f.add_face("kirk", "path/to/kirk.jpg")

# Swap faces
result = f2f.swap(target_image, faces="kirk", enhance_face_model="gpen_bfr_512")
```

## Customization Opportunities

Now that the code is local, you can:

1. **Simplify the API**: Remove unused features (video swapping, recognition, etc.)
2. **Optimize for Discord**: Pre-load models, add caching
3. **Add custom enhancements**: Implement custom face processing
4. **Improve error handling**: Better error messages for Discord context
5. **Add batch processing**: Process multiple images efficiently
6. **Custom face database**: Integrate with Discord user profiles
7. **Memory optimization**: Reduce memory footprint for long-running bot

## Dependencies

Required packages (already in pyproject.toml):
- `onnxruntime` - For running ONNX models
- `pillow` - Image processing
- `media-toolkit` - Media file handling (ImageFile, VideoFile)
- `opencv-python` (cv2) - Computer vision operations
- `numpy` - Numerical operations

## Models

Models are automatically downloaded from Azure Blob Storage on first use:
- Face detection: buffalo_l (InsightFace)
- Face swapping: inswapper_128
- Face enhancement: GPEN/GFPGAN variants

Model download location is configurable via `settings.py` (`MODELS_DIR`).

## Next Steps for Simplification

1. **Remove unused code**:
   - Video processing mixins (if not needed)
   - Face recognition features (if not needed)
   - Server functionality

2. **Consolidate imports**: Flatten the structure if you don't need modularity

3. **Add Discord-specific features**:
   - Progress callbacks for async operations
   - Better integration with discord.py
   - User-specific face storage

4. **Performance improvements**:
   - Model preloading
   - Result caching
   - Async/await optimizations

## Original Source

This code was extracted from: `socaity-face2face` version 1.3.0
- Repository: https://pypi.org/project/socaity-face2face/
- Based on InsightFace and similar face swapping technologies
