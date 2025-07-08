
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QGraphicsDropShadowEffect, QSpacerItem, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QObject
from PyQt5.QtGui import QFont, QPixmap, QIcon, QColor, QPalette, QLinearGradient
import torch
import time

class UtilityFunctions():
    
    @staticmethod
    def loadStyles(filename):
        """Load and apply a QSS stylesheet"""
        with open(filename, "r") as f:
            return f.read()
        
    @staticmethod
    def _set_sporty_background(self):
        """Set a dynamic sporty gradient background"""
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#0F2027"))  # Dark blue
        gradient.setColorAt(0.5, QColor("#203A43"))  # Medium blue
        gradient.setColorAt(1, QColor("#2C5364"))  # Light blue
        
        palette = self.palette()
        palette.setBrush(QPalette.Window, gradient)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
    
    @staticmethod
    def _setup_button_animations(self):
        """Setup animations for all buttons"""
        for btn in [self.btn_realtime, self.btn_local]:
            # Scale animation
            anim = QPropertyAnimation(btn, b"geometry")
            anim.setDuration(300)
            anim.setEasingCurve(QEasingCurve.OutBack)
            
            # Store animation on button object
            btn.animation = anim
            
            # Connect hover events
            btn.enterEvent = lambda e, b=btn: UtilityFunctions._animate_button(b, 5)
            btn.leaveEvent = lambda e, b=btn: UtilityFunctions._animate_button(b, 0)
    
    @staticmethod
    def _animate_button(button, offset):
        """Animate button on hover"""
        anim = button.animation
        geom = button.geometry()
        
        anim.stop()
        anim.setStartValue(geom)
        
        if offset > 0:  # Hover in
            geom = geom.adjusted(-offset, -offset, offset, offset)
        else:  # Hover out
            geom = geom.adjusted(5, 5, -5, -5)  # Return to normal
            
        anim.setEndValue(geom)
        anim.start()

class ModelWarmupWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._is_running = True

    def run(self):
        try:
            # Warmup in chunks to allow progress updates
            for i in range(10):
                if not self._is_running:
                    return
                
                # Do a portion of warmup work
                dummy = torch.randn(1, 3, 640, 640)
                if torch.cuda.is_available():
                    dummy = dummy.cuda()
                self.model(dummy)
                
                # Update progress (0-100)
                self.progress.emit((i+1)*10)
                time.sleep(0.1)  # Small delay to prevent free

            """
            dummy = torch.zeros((1, 3, 640, 640))
            
            if torch.cuda.is_available():
                self.progress.emit(20)
                dummy = dummy.cuda()
                self.model = self.model.cuda()
            
            self.progress.emit(30)
            with torch.no_grad():
                _ = self.model(dummy)
            
            self.progress.emit(100)
            
            for i in range(10):  # Progress steps
                self.model(dummy)
                self.progress.emit((i+1)*10)
            """
                
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False

    """
        def cleanup(self):
        self.deleteLater()
    """