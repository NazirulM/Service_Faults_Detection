import sys
import cv2
import numpy as np
import time
import random # not needed in the future
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout,
                            QGroupBox, QSplitter, QSlider)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from video_capture import VideoCaptureHandler as video_capture_handler

def check_available_cameras(max_to_check=3):
    available_cameras = []
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()  # Important: release the capture
    return available_cameras

available_cams = check_available_cameras()
print(f"Available cameras at indices: {available_cams}")

class TournamentDashboard(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set window properties
        self.setWindowTitle("AutoServo - Tournament Dashboard")
        self.setMinimumSize(1200, 800)
        self.setWindowIcon(QIcon("LogoFyp.png"))

        self.fault_counts = {
            "height_violation": 0,
            "foot_violation": 0,
            "double_racket_motion": 0,
            "delayed_racket_motion": 0
        }
        
        self.current_faults = {
            "height_violation": False,
            "foot_violation": False,
            "double_racket_motion": True,
            "delayed_racket_motion": False
        }
        
        # Setup UI
        self.setup_ui()
        
        # Initialize video capture
        self.video_handler = video_capture_handler(camera_index=1)  # Use default camera

        # FPS tracking variables
        self.frame_count = 0
        self.fps = 0
        self.fps_update_time = time.time()
        self.fps_history = []
        
        # Setup timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(16)  # Update every 100ms (related to fps)
        
        # Setup timer for fault detection simulation
        self.fault_timer = QTimer()
        self.fault_timer.timeout.connect(self.simulate_fault_detection)
        self.fault_timer.start(3000)  # Update every 3 seconds
    
    def setup_ui(self):
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        
        # Left panel (video and fault detection)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Header with logo and title
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        # In a real app, you would set an actual icon
        logo_label.setText("🏸")
        logo_label.setStyleSheet("font-size: 24px; background-color: #22c55e; color: white; border-radius: 15px; padding: 5px;")
        title_label = QLabel("AutoServo")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #166534;")
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Video feed
        video_group = QGroupBox("Live Video Feed")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background-color: #1f2937; color: white; border-radius: 8px;")
        self.video_label.setText("Camera Feed\n(Initializing...)")
        video_layout.addWidget(self.video_label)
        
        left_layout.addWidget(video_group)
        
        # Fault detection grid
        fault_grid = QGridLayout()
        
        # Height violation
        self.height_violation_box = self.create_fault_box("Height Violation")
        fault_grid.addWidget(self.height_violation_box, 0, 0)
        
        # Foot violation
        self.foot_violation_box = self.create_fault_box("Foot Violation")
        fault_grid.addWidget(self.foot_violation_box, 0, 1)
        
        # Double racket motion
        self.double_racket_box = self.create_fault_box("Double Racket Motion")
        fault_grid.addWidget(self.double_racket_box, 1, 0)
        
        # Delayed racket motion
        self.delayed_racket_box = self.create_fault_box("Delayed Racket Motion")
        fault_grid.addWidget(self.delayed_racket_box, 1, 1)
        
        left_layout.addLayout(fault_grid)
        
        # Right panel (confidence slider and fault summary)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Confidence Slider Group
        confidence_slider_group = QGroupBox("Confidence Threshold")
        confidence_slider_layout = QVBoxLayout(confidence_slider_group)

        # Create slider and label
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)  # 0-100%
        self.confidence_slider.setValue(50)      # Default threshold (70%)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)

        # Current value display
        self.confidence_value = QLabel("50%")
        self.confidence_value.setAlignment(Qt.AlignCenter)
        self.confidence_value.setStyleSheet("font-size: 16px; font-weight: bold;")

        # Description label
        confidence_desc = QLabel("Adjust ML model confidence threshold:")
        confidence_desc.setStyleSheet("font-size: 12px; color: #6b7280;")

        # Add widgets to layout
        confidence_slider_layout.addWidget(confidence_desc)
        confidence_slider_layout.addWidget(self.confidence_slider)
        confidence_slider_layout.addWidget(self.confidence_value)

        # Connect slider signal
        self.confidence_slider.valueChanged.connect(self.update_confidence_display)

        # Add to right panel
        right_layout.addWidget(confidence_slider_group)
        
        # Fault summary
        fault_summary_group = QGroupBox("Fault Summary")
        fault_summary_layout = QVBoxLayout(fault_summary_group)
        
        # Height violation count
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height Violation"))
        self.height_count = QLabel(str(self.fault_counts["height_violation"]))
        self.height_count.setStyleSheet("font-weight: bold;")
        height_layout.addStretch()
        height_layout.addWidget(self.height_count)
        fault_summary_layout.addLayout(height_layout)
        
        # Foot violation count
        foot_layout = QHBoxLayout()
        foot_layout.addWidget(QLabel("Foot Violation"))
        self.foot_count = QLabel(str(self.fault_counts["foot_violation"]))
        self.foot_count.setStyleSheet("font-weight: bold;")
        foot_layout.addStretch()
        foot_layout.addWidget(self.foot_count)
        fault_summary_layout.addLayout(foot_layout)
        
        # Double racket motion count
        double_layout = QHBoxLayout()
        double_layout.addWidget(QLabel("Double Racket Motion"))
        self.double_count = QLabel(str(self.fault_counts["double_racket_motion"]))
        self.double_count.setStyleSheet("font-weight: bold;")
        double_layout.addStretch()
        double_layout.addWidget(self.double_count)
        fault_summary_layout.addLayout(double_layout)
        
        # Delayed racket motion count
        delayed_layout = QHBoxLayout()
        delayed_layout.addWidget(QLabel("Delayed Racket Motion"))
        self.delayed_count = QLabel(str(self.fault_counts["delayed_racket_motion"]))
        self.delayed_count.setStyleSheet("font-weight: bold;")
        delayed_layout.addStretch()
        delayed_layout.addWidget(self.delayed_count)
        fault_summary_layout.addLayout(delayed_layout)
        
        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        fault_summary_layout.addWidget(line)
        
        # Total faults
        total_layout = QHBoxLayout()
        total_label = QLabel("Total Faults")
        total_label.setStyleSheet("font-weight: bold;")
        self.total_count = QLabel(str(sum(self.fault_counts.values())))
        self.total_count.setStyleSheet("font-weight: bold;")
        total_layout.addWidget(total_label)
        total_layout.addStretch()
        total_layout.addWidget(self.total_count)
        fault_summary_layout.addLayout(total_layout)
        
        fault_summary_layout.addStretch()
        right_layout.addWidget(fault_summary_group)
        
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes (2:1 ratio)
        splitter.setSizes([800, 400])

    
    def create_fault_box(self, title):
        """Create a box for displaying fault status"""
        box = QGroupBox(title)
        box.setStyleSheet("QGroupBox { border: 2px solid #22c55e; border-radius: 8px; background-color: #dcfce7; }")
        
        layout = QVBoxLayout(box)
        status_label = QLabel("LEGAL")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #16a34a;")
        layout.addWidget(status_label)
        
        # Store the status label as a property of the box for easy access
        box.status_label = status_label
        
        return box
    
    def update_fault_box(self, box, is_fault):
        """Update the appearance of a fault box based on fault status"""
        if is_fault:
            box.setStyleSheet("QGroupBox { border: 2px solid #ef4444; border-radius: 8px; background-color: #fee2e2; }")
            box.status_label.setText("FAULT")
            box.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #dc2626;")
        else:
            box.setStyleSheet("QGroupBox { border: 2px solid #22c55e; border-radius: 8px; background-color: #dcfce7; }")
            box.status_label.setText("LEGAL")
            box.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #16a34a;")
    
    @pyqtSlot()
    def update_video(self):
        """Update the video feed from the camera"""
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB format (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert the frame to a QImage
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale the image to fit the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.video_label.setText("Camera Error")
    

    @pyqtSlot()
    def update_confidence_display(self):
        """Update the confidence threshold display when slider moves"""
        value = self.confidence_slider.value()
        self.confidence_value.setText(f"{value}%")
        
        # Optional: Change color based on value
        if value > 80:
            self.confidence_value.setStyleSheet("color: #dc2626; font-size: 16px; font-weight: bold;")
        elif value > 50:
            self.confidence_value.setStyleSheet("color: #f59e0b; font-size: 16px; font-weight: bold;")
        else:
            self.confidence_value.setStyleSheet("color: #16a34a; font-size: 16px; font-weight: bold;")
        
        # In a real app, you would update your ML model's threshold here
        # self.ml_model.set_confidence_threshold(value/100)


    @pyqtSlot()
    def simulate_fault_detection(self):
        """Simulate fault detection (in a real app, this would be your ML model)"""
        # Generate random fault detections
        self.current_faults = {
            "height_violation": random.random() > 0.7,
            "foot_violation": random.random() > 0.8,
            "double_racket_motion": random.random() > 0.9,
            "delayed_racket_motion": random.random() > 0.85
        }
        
        # Update fault boxes
        self.update_fault_box(self.height_violation_box, self.current_faults["height_violation"])
        self.update_fault_box(self.foot_violation_box, self.current_faults["foot_violation"])
        self.update_fault_box(self.double_racket_box, self.current_faults["double_racket_motion"])
        self.update_fault_box(self.delayed_racket_box, self.current_faults["delayed_racket_motion"])
        
        # Update fault counts
        if self.current_faults["height_violation"]:
            self.fault_counts["height_violation"] += 1
            self.height_count.setText(str(self.fault_counts["height_violation"]))
        
        if self.current_faults["foot_violation"]:
            self.fault_counts["foot_violation"] += 1
            self.foot_count.setText(str(self.fault_counts["foot_violation"]))
        
        if self.current_faults["double_racket_motion"]:
            self.fault_counts["double_racket_motion"] += 1
            self.double_count.setText(str(self.fault_counts["double_racket_motion"]))
        
        if self.current_faults["delayed_racket_motion"]:
            self.fault_counts["delayed_racket_motion"] += 1
            self.delayed_count.setText(str(self.fault_counts["delayed_racket_motion"]))
        
        # Update total count
        self.total_count.setText(str(sum(self.fault_counts.values())))
    
    @pyqtSlot()
    def update_video(self):
        """Continuous video update loop"""
        frame = self.video_handler.read_frame()
        if frame is None:
            self.video_label.setText("Camera Feed Unavailable")
            return
            
        try:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Display the frame
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.width(), 
                    self.video_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
            
        except Exception as e:
            print(f"Error processing video frame: {str(e)}")
            self.video_label.setText("Video Processing Error")

    def closeEvent(self, event):
        """Clean up resources when the window is closed"""
        self.timer.stop()
        self.fault_timer.stop()

        # Release video handler resources instead of direct cap
        if hasattr(self, 'video_handler'):
            self.video_handler.release()

        # self.cap.release()

        event.accept()

