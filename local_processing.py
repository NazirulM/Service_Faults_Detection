import cv2
from PyQt5.QtWidgets import (
    QStackedWidget, QWidget, QLabel, QPushButton, QVBoxLayout, QGroupBox, QSlider,
    QHBoxLayout, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsRectItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QDragEnterEvent, QDropEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from utils.utility import UtilityFunctions
import time
import numpy as np
from collections import deque
from video_playback_widget import VideoPlaybackWidget


class LocalProcessing(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setupUI()
        
    def setupUI(self):
        # Main stacked widget for switching views
        self.inner_stack = QStackedWidget(self)  # CHANGED: Added 'self' as parent
        layout = QVBoxLayout(self)
        layout.addWidget(self.inner_stack)
        
        # Create both widgets
        self.upload_widget = FileUploadWindow(self)
        self.video_widget = VideoPlaybackWidget(self)

        # Set sporty gradient background
        UtilityFunctions._set_sporty_background(self)
        
        # Add to stack
        self.inner_stack.addWidget(self.upload_widget)
        self.inner_stack.addWidget(self.video_widget)
        
        # Connect signals
        self.upload_widget.fileSelected.connect(self.show_video_player)

    def show_video_player(self, file_path):
        self.video_widget.load_video(file_path)
        self.inner_stack.setCurrentWidget(self.video_widget)
        print(f"Current widget: {self.inner_stack.currentIndex()}")
    
    def go_back_to_upload(self):
        self.inner_stack.setCurrentWidget(self.upload_widget)


class FileUploadWindow(QWidget):
    fileSelected = pyqtSignal(str)  # Signal when file is selected
    
    def __init__(self, local_processing):
        super().__init__()
        self.local_processing = local_processing
        self.setupUI()
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        
        # Back button
        back_btn = QPushButton("← Back to Main Menu")
        back_btn.clicked.connect(self.goto_main_menu)
        layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        
        # Upload area
        upload_group = QGroupBox("Media File Input")
        upload_group.setObjectName("box_title")
        upload_layout = QVBoxLayout(upload_group)
        
        self.drop_area = QLabel("📁 Drag & drop files or Browse")
        self.drop_area.setObjectName("drop_area_label")
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.drop_area.setMinimumSize(400, 100)
        
        browse_btn = QPushButton("Browse Files")
        browse_btn.clicked.connect(self.browse_files)
        
        upload_layout.addWidget(self.drop_area)
        upload_layout.addWidget(browse_btn)
        layout.addWidget(upload_group)
        
        self.setAcceptDrops(True)

        # Drag and drop functionality
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_area.setStyleSheet("""
                QLabel {
                    border: 2px dashed #4CAF50;
                    background-color: #f8f8f8;
                    color: #4CAF50;
                }
            """)

    def dragLeaveEvent(self, event):
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                color: #777;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                color: #777;
            }
        """)
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        self.handle_files(files)

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Files", 
            "", 
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if files:
            self.handle_files(files)

    def handle_files(self, files):
        # Process your files here
        print("Selected files:", files)
        self.drop_area.setText(f"{len(files)} file(s) selected")
        
        # Example: Load first video file
        if files:
            self.fileSelected.emit(files[0])  # Emit first selected file
            # Here you would actually load/process the video

    def goto_main_menu(self):
        """Return to main menu through main window's stack"""
        self.local_processing.main_window.stack.setCurrentIndex(0)


"""
class VideoPlaybackWidget(QWidget):
    def __init__(self, local_processing):
        super().__init__()
        self.local_processing = local_processing
        self.video_path = ""
        self.cap = None
        self.timer = QTimer()
        self.current_frame = None
        self.fps = 50
        self.frame_history = deque(maxlen=50)  # Store last 30 frames for chart
        self.violation_counts = {'height': 0, 'foot': 0}
        self.setupUI()

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
        
        # Violation Summary
        summary_group = QGroupBox("Service Faults Summary")
        summary_group.setObjectName("box_title")
        summary_layout = QVBoxLayout(summary_group)
        
        self.height_violation_label = QLabel("Height Violations: 0")
        self.foot_violation_label = QLabel("Foot Violations: 0")
        self.height_violation_label.setObjectName("description_label")
        self.foot_violation_label.setObjectName("description_label")

        summary_layout.addWidget(self.height_violation_label)
        summary_layout.addWidget(self.foot_violation_label)
        right_layout.addWidget(summary_group)
        
        # Fault/Legal Dashboard
        self.status_dashboard = QLabel()
        self.status_dashboard.setAlignment(Qt.AlignCenter)
        self.status_dashboard.setStyleSheet(
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                border-radius: 10px;
            }
        )
        self.update_status_display()
        right_layout.addWidget(self.status_dashboard)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)

    def load_video(self, file_path):
        self.video_path = file_path
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("Error opening video file")
            return
        
            
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // self.fps)  # 60 FPS
        
        # Reset analysis data
        self.violation_counts = {'height': 0, 'foot': 0}
        self.frame_history.clear()
        self.update_violation_display()
        self.update_status_display()

    def update_frame(self):
        
        if not hasattr(self, 'fps_start_time'):
            self.fps_start_time = time.time()
            self.fps_frame_count = 0

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
            
        # Process frame (example - replace with your actual processing)
        processed_frame, confidence = self.process_frame(frame)
        
        # Convert to QImage and display
        h, w, ch = processed_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(q_img))
        
        # Update chart data
        self.frame_history.append(confidence)
        self.update_chart()

        # 5. FPS Measurement and Display ----------------------
        self.fps_frame_count += 1
        
        # Calculate every second
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:  # Update FPS every 1 second
            self.measured_fps = self.fps_frame_count / elapsed
            print(f"Current FPS: {self.measured_fps:.1f}")
            
            # Optional: Update a QLabel to show FPS on UI
            if hasattr(self, 'fps_label'):
                self.fps_label.setText(f"FPS: {self.measured_fps:.1f}")
            
            # Reset counters
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        
        
        # Simulate violations (replace with your actual detection)
        if np.random.random() < 0.05:  # 5% chance of violation
            if np.random.random() < 0.5:
                self.violation_counts['height'] += 1
            else:
                self.violation_counts['foot'] += 1
            self.update_violation_display()
            self.update_status_display()
        
        

    def process_frame(self, frame):
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Example processing - replace with your actual analysis
        # Here we just return a random confidence value for demonstration
        confidence = np.random.randint(0, 100)
        
        # Draw some example annotations
        cv2.putText(frame, f"Confidence: {confidence}%", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame, confidence

    def update_chart(self):
        self.line.set_data(range(len(self.frame_history)), list(self.frame_history))
        self.ax.set_xlim(0, len(self.frame_history))
        self.canvas.draw()

    def update_threshold(self, value):
        self.threshold_label.setText(f"Current Threshold: {value}%")
        # Add your threshold-based processing here

    def update_violation_display(self):
        self.height_violation_label.setText(f"Height Violations: {self.violation_counts['height']}")
        self.foot_violation_label.setText(f"Foot Violations: {self.violation_counts['foot']}")

    def update_status_display(self):
        if self.violation_counts['height'] > 0 or self.violation_counts['foot'] > 0:
            self.status_dashboard.setText("FAULT DETECTED")
            self.status_dashboard.setStyleSheet("background-color: #ffcccc; color: #cc0000;")
        else:
            self.status_dashboard.setText("LEGAL SERVICE")
            self.status_dashboard.setStyleSheet("background-color: #ccffcc; color: #00aa00;")

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.timer.isActive():
            self.timer.stop()
        event.accept()
"""