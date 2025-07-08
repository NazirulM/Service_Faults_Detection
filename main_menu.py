from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QGraphicsDropShadowEffect, QSpacerItem, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QSize, QPoint
from PyQt5.QtGui import QFont, QPixmap, QIcon, QColor, QPalette, QLinearGradient
from utils.utility import UtilityFunctions

class MainMenu(QWidget):

    realtime_clicked = pyqtSignal()
    local_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # Add top spacer
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.welcome_label = QLabel("AutoServo")
        self.subtitle_label = QLabel("Play Fair")
        self.welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.welcome_label.setObjectName("welcome_label")
        self.subtitle_label.setObjectName("subtitle_label")

        # Button container
        button_container = QVBoxLayout()
        button_container.setSpacing(25)

        # Real-time Processing button with description
        self.btn_realtime = self._create_sporty_button(
            "REAL-TIME PROCESSING", 
            "Start your webcam and challenge the system in real time!", 
            "#2E86AB",  # Blue color
            
        )
        
        # Local Processing button with description
        self.btn_local = self._create_sporty_button(
            "LOCAL PROCESSING", 
            "Upload recorded videos for detailed analysis and feedback", 
            "#8dd9cc",  # blue-green color
            
        )

        # self.btn_realtime.setObjectName("realtime_button")
        # self.btn_local.setObjectName("local_button")

        # Add widgets to layouts
        button_container.addWidget(self.btn_realtime)
        button_container.addWidget(self.btn_local)
        
        layout.addWidget(self.welcome_label)
        layout.addWidget(self.subtitle_label)
        layout.addLayout(button_container)
        
        # Add bottom spacer
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.setLayout(layout)
        
        # Set sporty gradient background
        UtilityFunctions._set_sporty_background(self)
        
        # Add hover animations
        UtilityFunctions._setup_button_animations(self)

    
    def _create_sporty_button(self, title, description, base_color):

        button_frame = QFrame()
        button_frame.setStyleSheet(f"""
            background-color: {base_color};
            border-radius: 10px;
            border: 2px solid {base_color};
        }}
        QFrame:hover {{
            background-color: transparent;
            border: 2px solid {base_color};
        }}
        """)

        # Main button layout
        btn_layout = QVBoxLayout(button_frame)
        btn_layout.setContentsMargins(15, 15, 15, 15)
        btn_layout.setSpacing(8)
        
        # Title with icon
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(15)
            
        # Button title
        title_label = QLabel(title)
        title_label.setObjectName("title_label")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        # Description
        desc_label = QLabel(description)
        desc_label.setObjectName("description_label")
        desc_label.setWordWrap(True)
        
        # Add to main button layout
        btn_layout.addLayout(title_layout)
        btn_layout.addWidget(desc_label)
        
        # Change description color on hover
        button_frame.enterEvent = lambda e: (
            desc_label.setStyleSheet(f"QLabel {{ color: {base_color}; font-size: 14px; }}"),
            title_label.setStyleSheet(f"QLabel {{ color: {base_color}; font-size: 18px; font-weight: bold; }}")
        )
        button_frame.leaveEvent = lambda e: (
            desc_label.setStyleSheet("QLabel { color: white; font-size: 14px; }"),
            title_label.setStyleSheet("QLabel { color: white; font-size: 18px; font-weight: bold; }")
        )
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(base_color).darker(120))
        shadow.setOffset(4, 4)
        button_frame.setGraphicsEffect(shadow)
        
        # Make the whole frame clickable
        button_frame.mousePressEvent = lambda e: self._handle_button_click(title)
        button_frame.setCursor(Qt.PointingHandCursor)
        
        return button_frame
        
    def _handle_button_click(self, button_title):
        if button_title == "REAL-TIME PROCESSING":
            self.realtime_clicked.emit()
        elif button_title == "LOCAL PROCESSING":
            self.local_clicked.emit()