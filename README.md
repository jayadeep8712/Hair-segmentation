# Hair Segmentation with MediaPipe  

## Overview  
This repository utilizes Google's MediaPipe Image Segmenter to detect and segment hair from an image. The model identifies the hair region and provides an image segmentation mask, which can be used for hair recoloring, styling, or effects.  

## Model & Resources  
- **Official Model:** [Hair Segmentation Model](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)  
- **Download Model:** [hair_segmenter.tflite](https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite)  

## Model Details  
This segmentation model classifies each pixel in an image into two categories:  
| Class | Description |
|-------|------------|
| `0` | Background (non-hair areas) |
| `1` | Hair (detected hair regions) |

## Features & Use Cases  
- Hair recoloring for style visualization  
- Virtual hair effects (filters, animations)  
- Pre-processing for hairstyle recommendations  

## Installation & Usage  

### 1. Set up a virtual environment  
It is recommended to use a **virtual environment** to avoid dependency conflicts.  

#### On Windows (Command Prompt or PowerShell):  
```bash
python -m venv venv
venv\Scripts\activate
```
#### On macOS/Linux (Terminal):  
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Clone this repository  
```bash
git clone https://github.com/jayadeep8712/Hair-segmentation.git  
cd hair-segmentation  
```

### 3. Install dependencies  
```bash
pip install mediapipe opencv-python numpy  
```

### 4. Run the hair segmentation script  
```python
import cv2
import mediapipe as mp
```
*For a full implementation, check the script in* `hair_segmentation.py`  

## Example Output  
| Input Image | Hair Segmentation Mask |  
|-------------|------------------------|  
| ![Original](image.png) | ![Segmented](mask.png) |  

## License  
This project follows the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for details.  

## Notes  
- This model works best on well-lit images with clear hair visibility.  
- For real-time hair segmentation, consider using MediaPipe in a webcam pipeline.  
 