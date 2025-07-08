from PyQt5.QtWidgets import (
    QStackedWidget, QWidget, QLabel, QPushButton, QVBoxLayout, 
    QGroupBox, QHBoxLayout, QFileDialog, QSlider,
    QProgressDialog, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThreadPool, QThread
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QDragEnterEvent, QDropEvent
from utils.utility import ModelWarmupWorker
from video_processer import FrameProcessorSignals, FrameProcessor
import time
import cv2
from collections import deque
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import torch
import numpy as np
from ultralytics import YOLO
import torch


class VideoPlaybackWidget(QWidget):
    def __init__(self, local_processing):
        super().__init__()
        self.local_processing = local_processing
        self.video_path = ""
        self.cap = None
        self.timer = QTimer()
        self.current_frame = None
        self.fps = 30
        self.frame_history = deque(maxlen=50)  # Store last 50 frames for chart

        # Enhanced GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load YOLOv8 model
        self.model = YOLO("runs/detect/yolov8n_custom/weights/best.pt")  # Make sure this path is correct
        self.model.to(self.device)

        if torch.cuda.is_available():
            #self.model = self.model.cuda()
            torch.backends.cudnn.benchmark = True
        
        # Thread pool setup
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)  # Adjust based on your hardware
        self.pending_frames = 0
        self.last_processed_frame = None

        self.setupUI()
        
        # Video info tracking
        self.measured_fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.bounding_box_count = 0  # Will be updated by your model

        # Thread management
        self.warmup_thread = None  # Track our warmup thread
        self.warmup_worker = None
        self.loading_dialog = None

        # Remove the dialog creation from __init__
        self.loading_dialog = None  # We'll create it when needed

        # Model warmup flag
        self.model_ready = False


    def setupUI(self):
        # Main horizontal layout for left and right panels
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Left Panel (Video Playback)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Back button
        back_btn = QPushButton("← Back to Upload")
        back_btn.clicked.connect(self.local_processing.go_back_to_upload)
        left_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        
        # Video display
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 360)
        self.video_display.setStyleSheet("background-color: black; color: white;")
        left_layout.addWidget(self.video_display, stretch=1)
        
        # Right Panel (Analysis Dashboard)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Confidence Threshold Slider
        threshold_group = QGroupBox("Confidence Threshold")
        threshold_group.setObjectName("box_title")
        threshold_layout = QVBoxLayout(threshold_group)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(35)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel(f"Current Threshold: 35%")
        self.threshold_label.setObjectName("description_label")
        threshold_layout.addWidget(self.threshold_label)
        right_layout.addWidget(threshold_group)
        
        # Matplotlib Figure for Line Chart
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Confidence Over Time")
        self.ax.set_xlabel("Frames")
        self.ax.set_ylabel("Confidence %")
        self.ax.set_ylim(0, 100)
        self.line, = self.ax.plot([], [])
        self.canvas.setMinimumHeight(200)
        right_layout.addWidget(self.canvas)
        
        # Video Information Panel (replaces Violation Summary)
        info_group = QGroupBox("Video Information")
        info_group.setObjectName("box_title")
        info_layout = QVBoxLayout(info_group)
        
        # FPS display
        self.fps_label = QLabel("Playback FPS: 0.0")
        self.fps_label.setObjectName("description_label")
        
        # Bounding box count display
        self.bbox_label = QLabel("Bounding Boxes: 0")
        self.bbox_label.setObjectName("description_label")
        
        # Video resolution display
        self.resolution_label = QLabel("Resolution: N/A")
        self.resolution_label.setObjectName("description_label")
        
        # Frame count display
        self.frame_count_label = QLabel("Frame: 0")
        self.frame_count_label.setObjectName("description_label")
        
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.bbox_label)
        info_layout.addWidget(self.resolution_label)
        info_layout.addWidget(self.frame_count_label)
        right_layout.addWidget(info_group)
        
        # Status Dashboard (can keep this or modify as needed)
        self.status_dashboard = QLabel()
        self.status_dashboard.setAlignment(Qt.AlignCenter)
        self.status_dashboard.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                border-radius: 10px;
            }
        """)
        self.status_dashboard.setText("LEGAL SERVICE")
        self.status_dashboard.setStyleSheet("background-color: #e6f3ff; color: #0066cc;")
        right_layout.addWidget(self.status_dashboard)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)

    def load_video(self, file_path):
        self.video_path = file_path

        if self.cap:
            self.cap.release()

        # Initialize dialog with proper parent and settings
        self.loading_dialog = QProgressDialog(
            "Initializing model...", 
            "Cancel", 
            0, 
            100, 
            None  # Critical parent parameter
        )
        self.loading_dialog.setWindowTitle("Processing Status")
        self.loading_dialog.setWindowModality(Qt.WindowModal)
        #self.loading_dialog.setAttribute(Qt.WA_DeleteOnClose, False)  # Prevent auto-deletion
        #self.loading_dialog.canceled.connect(self.handle_cancel)  # Proper signal connection
        self.loading_dialog.setCancelButton(None)  # Disable cancel button initially
        self.loading_dialog.setAutoClose(False)
        self.loading_dialog.setAutoReset(False)

        # Show loading dialog before processing starts
        self.loading_dialog.setLabelText("Loading video and initializing model...")
        self.loading_dialog.setValue(0)
        self.loading_dialog.show()

        # Properly manage warmup thread
        if self.warmup_thread and self.warmup_thread.isRunning():
            self.warmup_thread.quit()

        # Warm up model in background
        self.warmup_model()
        
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("Error opening video file")
            return
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution_label.setText(f"Resolution: {width}x{height}")
        
        # Reset counters
        self.frame_count = 0
        self.start_time = time.time()
        self.bounding_box_count = 0
        self.update_info_display()
            
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // self.fps)  # 50 FPS
        
        # Reset frame history
        self.frame_history.clear()
    
    def warmup_model(self):
         
        if hasattr(self, 'warmup_thread') and self.warmup_thread:
            if hasattr(self, 'warmup_worker') and self.warmup_worker:
                self.warmup_worker.stop()  # Signal worker to stop
            # self.warmup_thread.quit()

            self.cleanup_warmup()

            
        # Run in a separate thread to keep UI responsive
        self.warmup_thread = QThread()
        self.warmup_worker = ModelWarmupWorker(self.model)
        self.warmup_worker.moveToThread(self.warmup_thread)
        
        self.warmup_worker.finished.connect(self.on_warmup_complete)
        self.warmup_worker.error.connect(self.on_warmup_error)
        self.warmup_worker.progress.connect(self.loading_dialog.setValue)
        self.warmup_thread.started.connect(self.warmup_worker.run)
        self.warmup_thread.start()

        # Run dummy inference to initialize model
        if self.loading_dialog:
            self.loading_dialog.setLabelText("Warming up model...")
            # Enable cancel button only after dialog is shown
            self.loading_dialog.setCancelButtonText("Cancel")
            self.loading_dialog.canceled.connect(self.handle_cancel)    

    def cleanup_warmup(self):
        """Safely clean up warmup resources"""
        if hasattr(self, 'warmup_worker') and self.warmup_worker:
            self.warmup_worker.stop()
        
        if hasattr(self, 'warmup_thread') and self.warmup_thread:
            if self.warmup_thread.isRunning():
                self.warmup_thread.quit()
                # Give it a short time to finish, but don't block
                # QTimer.singleShot(300, self.warmup_thread.deleteLater)
                self.warmup_thread.wait(1000)
            else:
                self.warmup_thread.deleteLater()
        
        # Clear references
        if hasattr(self, 'warmup_worker'):
            del self.warmup_worker
        if hasattr(self, 'warmup_thread'):
            del self.warmup_thread

    
    def cancel_loading(self):
        """Proper cancellation handling"""
        if self.warmup_thread and self.warmup_thread.isRunning():
            self.warmup_thread.quit()
        
        if self.loading_dialog:
            self.loading_dialog.close()
            self.loading_dialog = None
            
        # Optional: go back to upload screen
        self.local_processing.go_back_to_upload()

    def on_warmup_complete(self):
        self.model_ready = True
        self.loading_dialog.setValue(100)
        self.loading_dialog.setLabelText("Model ready! Starting video processing...")
        # Only close the dialog, not the parent widget
        QTimer.singleShot(1000, self.loading_dialog.hide)

    def on_warmup_error(self, error_msg):
        self.loading_dialog.cancel()
        QMessageBox.critical(self, "Model Error", f"Failed to initialize model:\n{error_msg}")
    
    def cancel_processing(self):
        """Handle user cancellation"""
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.model_ready = False
        self.loading_dialog.cancel()

    def safe_close_dialog(self):
        if self.loading_dialog:
            self.loading_dialog.hide()
            # self.loading_dialog.deleteLater()
            self.loading_dialog = None
    
    def handle_cancel(self):
        """Safe cancellation handler"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        self.safe_close_dialog()

    def update_frame(self):

        if not self.model_ready:
            return
        
        # Skip frame if queue is full
        if self.pending_frames >= self.threadpool.maxThreadCount():
            if self.last_processed_frame:
                self.display_frame(*self.last_processed_frame)
            return
        

        # Only process if threadpool has available threads
        if self.threadpool.activeThreadCount() < self.threadpool.maxThreadCount():
            ret, frame = self.cap.read()
            if not ret:
                self.frame_count = 0    
                self.pending_frames = 0
                # Video ended - reset to start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.timer.stop()
                    if hasattr(self, 'loading_dialog') and self.loading_dialog:
                        self.loading_dialog.hide()
                    return
            
            # Create and start processing task
            processor = FrameProcessor(frame, self.model, 
                                    self.threshold_slider.value(),
                                    self.device, img_size=640)
            processor.signals.finished.connect(self.on_processing_finished)
            processor.signals.error.connect(self.on_processing_error)
            self.threadpool.start(processor)
            
            self.frame_count += 1
            self.pending_frames += 1
            
            
            # Update info display
            self.update_info_display()
   

    def on_processing_finished(self, processed_frame, confidence, bbox_count):
        self.pending_frames -= 1
        self.bounding_box_count = bbox_count
        self.last_processed_frame = (processed_frame, confidence, bbox_count)
        self.display_frame(processed_frame, confidence, bbox_count)

    def on_processing_error(self, error_msg):
        self.pending_frames -= 1
        print(f"Frame processing error: {error_msg}")
        self.loading_dialog.setLabelText(f"Error: {error_msg}")
        QTimer.singleShot(2000, lambda: self.loading_dialog.setLabelText("Resuming processing..."))

    def display_frame(self, processed_frame, confidence, bbox_count):
        # Update frame history for chart
        self.frame_history.append(confidence)
        self.update_chart()
        
        # Convert to QImage and display
        h, w, ch = processed_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(q_img))


    def process_frame(self, frame):
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Example processing - replace with your actual analysis
        # Here we just return random values for demonstration
        confidence = np.random.randint(0, 100)
        bbox_count = np.random.randint(0, 5)  # Random number of boxes (0-4)
        
        # Draw some example annotations
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Boxes: {bbox_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw example bounding boxes (random positions)
        for i in range(bbox_count):
            x1, y1 = np.random.randint(0, frame.shape[1]//2), np.random.randint(0, frame.shape[0]//2)
            x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(50, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame, confidence, bbox_count


    def update_info_display(self):
        # Calculate FPS
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.measured_fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {self.measured_fps:.1f}")
        
        # Update other info
        self.bbox_label.setText(f"Bounding Boxes: {self.bounding_box_count}")
        self.frame_count_label.setText(f"Frame: {self.frame_count}")

    def update_chart(self):
        self.line.set_data(range(len(self.frame_history)), list(self.frame_history))
        self.ax.set_xlim(0, len(self.frame_history))
        self.canvas.draw()

    def update_threshold(self, value):
        self.threshold_label.setText(f"Current Threshold: {value}%")

    def closeEvent(self, event):
        if self.warmup_thread and self.warmup_thread.isRunning():
            self.warmup_thread.quit()
            self.warmup_thread.wait(1000)

        if self.cap and self.cap.isOpened():
            self.cap.release()

        if self.timer.isActive():
            self.timer.stop()

        self.safe_close_dialog()
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Additional cleanup for shared memory

        # Clean up warmup resources
        self.cleanup_warmup()

        event.accept()

