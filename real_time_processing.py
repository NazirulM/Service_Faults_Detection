import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QGroupBox, QHBoxLayout, QSlider
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from utils.utility import UtilityFunctions
import time

class RealTimeProcessing(QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.cap = None
        self.last_inference_time = 0
        self.frame_count = 0
        self.fps = 0
        self.inference_speed = 0
        self.confidence = 50

        self.main_window = main_window

        self.setWindowTitle("AutoServo")
        self.setMinimumSize(1000, 800)
        self.setWindowIcon(QIcon("LogoFyp.png"))
        UtilityFunctions._set_sporty_background(self)

        main_layout = QVBoxLayout()

        # Back button
        back_button = QPushButton("← Back to Main Menu")
        back_button.setObjectName("back_button")
        back_button.clicked.connect(self.goto_main_menu)  # Connect to new method
        main_layout.addWidget(back_button)


        # Video Stream Panel
        self.video_label = QLabel("Webcam Feed\n(Not started)")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 320)
        self.video_label.setStyleSheet("background-color: #222; color: #fff; border-radius: 8px;")
        main_layout.addWidget(self.video_label)

        # Fault Dashboards
        fault_layout = QHBoxLayout()
        self.height_fault = self.create_fault_box("Height Violation")
        self.foot_fault = self.create_fault_box("Foot Violation")
        self.racket_fault = self.create_fault_box("Racket Movement Violation")
        fault_layout.addWidget(self.height_fault)
        fault_layout.addWidget(self.foot_fault)
        fault_layout.addWidget(self.racket_fault)
        main_layout.addLayout(fault_layout)

        # Confidence Slider
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence Level:")
        conf_label.setObjectName("realtime_labels")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(self.confidence)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        self.conf_value = QLabel(f"{self.confidence}%")
        self.conf_value.setObjectName("realtime_labels")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value)
        main_layout.addLayout(conf_layout)

        # Frame Rate and Inference Speed
        stats_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0")
        self.infer_label = QLabel("Inference: 0 ms")
        self.fps_label.setObjectName("realtime_labels")
        self.infer_label.setObjectName("realtime_labels")
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.infer_label)
        main_layout.addLayout(stats_layout)

        # Start Webcam Button
        self.start_btn = QPushButton("Start Webcam")
        self.start_btn.clicked.connect(self.start_webcam)
        main_layout.addWidget(self.start_btn)

        self.setLayout(main_layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
    
    def goto_main_menu(self):
        """Switch back to the main menu in the stacked widget."""
        if self.parent():  # Check if parent exists (stacked widget)
            self.parent().setCurrentIndex(0)  # Switch to first widget (main menu)

        # print(f"Current widget: {self.stack.currentIndex()}")

    def create_fault_box(self, title):
        box = QGroupBox(title)
        box.setObjectName("box_title")
        layout = QVBoxLayout()
        label = QLabel("LEGAL")
        label.setObjectName("realtime_labels")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 16px; font-weight: bold; color: #16a34a;")
        layout.addWidget(label)
        box.setLayout(layout)
        box.status_label = label
        return box

    def update_confidence(self, value):
        self.confidence = value
        self.conf_value.setText(f"{value}%")
        # You can update your ML model's confidence threshold here
        # if self.ml_model:
        #     self.ml_model.set_confidence_threshold(value / 100)

    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.video_label.setText("Webcam not accessible!")
                return
            self.start_btn.setEnabled(False)
            self.timer.start(30)
            self.last_inference_time = time.time()
            self.frame_count = 0

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.video_label.setText("No frame")
            return
        # --- ML Inference Placeholder ---
        start_inf = time.time()
        # Here you would run your ML model on the frame
        # Example: results = self.ml_model.infer(frame)
        # For now, we mock the results:
        import random
        height_fault = random.random() > 0.8
        foot_fault = random.random() > 0.85
        racket_fault = random.random() > 0.9
        # --- End ML Placeholder ---
        end_inf = time.time()
        self.inference_speed = int((end_inf - start_inf) * 1000)  # ms
        self.infer_label.setText(f"Inference: {self.inference_speed} ms")
        # Update fault boxes
        self.set_fault_status(self.height_fault, height_fault)
        self.set_fault_status(self.foot_fault, foot_fault)
        self.set_fault_status(self.racket_fault, racket_fault)
        # Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.copy().data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))
        # FPS calculation
        self.frame_count += 1
        now = time.time()
        if now - self.last_inference_time >= 1.0:
            self.fps = self.frame_count
            self.fps_label.setText(f"FPS: {self.fps}")
            self.frame_count = 0
            self.last_inference_time = now

    def set_fault_status(self, box, is_fault):
        if is_fault:
            box.status_label.setText("FAULT")
            box.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
        else:
            box.status_label.setText("LEGAL")
            box.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()