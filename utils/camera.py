import cv2

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