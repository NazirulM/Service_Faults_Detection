import cv2

class VideoCaptureHandler:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.frame = None
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize or reinitialize camera with error handling"""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Failed to open camera at index {self.camera_index}")
            return False
            
        # Set preferred camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        return True

    def read_frame(self):
        """Read next frame with error handling"""
        if not self.running or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            print("Frame capture failed - attempting to reinitialize camera")
            if not self.initialize_camera():
                self.running = False
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
                
        self.frame = frame
        return frame

    def release(self):
        """Release camera resources"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None