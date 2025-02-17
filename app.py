from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

app = Flask(__name__)

# Global variables
camera = None
hair_segmenter = None
is_segmenting = False
current_color = [255, 0, 0]  # Default red color
output_frame = None
last_timestamp = 0

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.is_running = False
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if success:
            return frame
        return None

class HairSegmentation:
    def __init__(self, model_path):
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
            output_confidence_masks=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        self.last_timestamp = 0

    def process_frame(self, frame, timestamp_ms, overlay_color):
        # Ensure timestamp is greater than the last one
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        segmentation_result = self.segmenter.segment_for_video(
            mp_image,
            timestamp_ms
        )
        
        category_mask = segmentation_result.category_mask.numpy_view()
        hair_mask = (category_mask == 1).astype(np.uint8) * 255
        
        overlay = frame.copy()
        overlay[hair_mask == 255] = overlay_color
        
        alpha = 0.5
        output = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return output

def get_timestamp():
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)

def generate_frames():
    global camera, hair_segmenter, is_segmenting, current_color, output_frame
    base_timestamp = get_timestamp()
    frame_count = 0
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
            
        if is_segmenting:
            try:
                # Calculate timestamp based on frame count and base timestamp
                current_timestamp = base_timestamp + (frame_count * 33)  # Assuming ~30fps
                frame = hair_segmenter.process_frame(frame, current_timestamp, current_color)
                frame_count += 1
            except Exception as e:
                print(f"Error in segmentation: {e}")
                # Reset timestamps if needed
                base_timestamp = get_timestamp()
                frame_count = 0
                
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        output_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_segmentation')
def toggle_segmentation():
    global is_segmenting
    is_segmenting = not is_segmenting
    return jsonify({'status': 'success', 'is_segmenting': is_segmenting})

@app.route('/update_color/<r>/<g>/<b>')
def update_color(r, g, b):
    global current_color
    current_color = [int(b), int(g), int(r)]  # BGR format for OpenCV
    return jsonify({'status': 'success', 'color': current_color})

def initialize():
    global camera, hair_segmenter
    camera = Camera()
    MODEL_PATH = r"E:\Hairsegflask\static\models\hair_segmenter.tflite"  # Update with your model path
    hair_segmenter = HairSegmentation(MODEL_PATH)

if __name__ == '__main__':
    initialize()
    app.run(port=5001,debug=True)