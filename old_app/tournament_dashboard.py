import sys
import cv2
import numpy as np
import random # not needed in the future
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout,
                            QGroupBox, QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSlot


class TournamentDashboard(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set window properties
        self.setWindowTitle("AutoServo - Tournament Dashboard")
        self.setMinimumSize(1200, 800)
        self.setWindowIcon(QIcon("LogoFyp.png"))

        
        # Initialize match data
        
        self.match_info = {
            "category": "Singles",
            "players": ["Player 1", "Player 2"],
            "status": "Active"
        }
        
        

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
        self.cap = cv2.VideoCapture(0)  # Use default camera
        
        # Setup timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(100)  # Update every 100ms (related to fps)
        
        # Setup timer for fault detection simulation
        self.fault_timer = QTimer()
        self.fault_timer.timeout.connect(self.simulate_fault_detection)
        self.fault_timer.start(3000)  # Update every 3 seconds
    
    def setup_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
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
        
        # Back button
        back_button = QPushButton("← Back to Setup")
        back_button.setStyleSheet("padding: 8px;")
        header_layout.addWidget(back_button)
        
        left_layout.addLayout(header_layout)
        
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

        
        
        # Right panel (fault summary)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Match information
        match_info_group = QGroupBox("Match Information")
        match_info_layout = QVBoxLayout(match_info_group)
        
        # Category
        category_label = QLabel("Category")
        category_label.setStyleSheet("font-size: 12px; color: #6b7280;")
        self.category_value = QLabel(self.match_info["category"])
        self.category_value.setStyleSheet("font-weight: bold;")
        match_info_layout.addWidget(category_label)
        match_info_layout.addWidget(self.category_value)
        match_info_layout.addSpacing(10)
        
        # Players
        players_label = QLabel("Players")
        players_label.setStyleSheet("font-size: 12px; color: #6b7280;")
        players_text = " vs ".join(self.match_info["players"])
        self.players_value = QLabel(players_text)
        self.players_value.setStyleSheet("font-weight: bold;")
        match_info_layout.addWidget(players_label)
        match_info_layout.addWidget(self.players_value)
        match_info_layout.addSpacing(10)
        
        # Status
        status_label = QLabel("Status")
        status_label.setStyleSheet("font-size: 12px; color: #6b7280;")
        self.status_value = QLabel(self.match_info["status"])
        self.status_value.setStyleSheet("font-weight: bold; color: #16a34a;")
        match_info_layout.addWidget(status_label)
        match_info_layout.addWidget(self.status_value)
        
        match_info_layout.addStretch()
        right_layout.addWidget(match_info_group)
        
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
    
    def closeEvent(self, event):
        """Clean up resources when the window is closed"""
        self.timer.stop()
        self.fault_timer.stop()
        self.cap.release()
        event.accept()

