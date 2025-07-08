from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QThreadPool, QTimer
from PyQt5.QtGui import QImage, QPixmap
from test_model import FrameProcessor
import cv2

class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.threadpool = QThreadPool()
        print(f"Max threads: {self.threadpool.maxThreadCount()}")  # Typically CPU cores
        
        # Setup UI
        self.setup_ui()
        
        # For video processing example
        self.cap = cv2.VideoCapture(0)  # Webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)  # ~30 FPS
        
    def setup_ui(self):
        self.label = QLabel()
        self.setCentralWidget(self.label)
        
    def process_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Create worker for this frame
            worker = FrameProcessor(frame, self.model)
            worker.signals.result.connect(self.display_result)
            worker.signals.error.connect(self.handle_error)
            
            # Start processing in thread pool
            self.threadpool.start(worker)
    
    def display_result(self, data):
        frame, results = data
        # Display processed frame or results
        processed_frame = self.draw_results(frame, results)
        self.label.setPixmap(self.array_to_pixmap(processed_frame))
    
    def draw_results(self, frame, results):
        # Draw bounding boxes, labels, etc.
        # Example for object detection:
        for box, label, score in results:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame
    
    def array_to_pixmap(self, frame):
        # Convert OpenCV BGR to RGB then to QPixmap
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)
    
    def handle_error(self, message):
        print(f"Error in processing: {message}")