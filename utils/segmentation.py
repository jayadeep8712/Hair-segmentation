import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

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

    def process_frame(self, frame, timestamp_ms, overlay_color):
        """
        Process a single frame for hair segmentation
        Args:
            frame: Input frame from camera/video
            timestamp_ms: Timestamp of the frame
        Returns:
            Original frame with overlay and segmentation mask
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Perform segmentation
        segmentation_result = self.segmenter.segment_for_video(
            mp_image,
            timestamp_ms
        )
        
        # Get category mask (hair is category 1)
        category_mask = segmentation_result.category_mask.numpy_view()
        
        # Create visualization
        height, width = frame.shape[:2]
        hair_mask = (category_mask == 1).astype(np.uint8) * 255
        
        # Create colored overlay for visualization
        overlay = frame.copy()
        overlay[hair_mask == 255] = [0, 0, 255]  # Red color for hair
        
        # Blend with original frame
        alpha = 0.5
        output = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return output, hair_mask

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize hair segmentation with the correct model path
    # Use raw string to avoid escape sequence issues
    MODEL_PATH = r"E:\Hairsegflask\static\models\hair_segmenter.tflite" 
    hair_segmenter = HairSegmentation(MODEL_PATH)
    
    timestamp = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        result, mask = hair_segmenter.process_frame(frame, timestamp)
        timestamp += 1  # Increment timestamp for each frame
        
        # Display results
        cv2.imshow('Hair Segmentation', result)
        cv2.imshow('Hair Mask', mask)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()