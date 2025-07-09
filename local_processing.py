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
