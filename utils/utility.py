
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QGraphicsDropShadowEffect, QSpacerItem, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QObject
from PyQt5.QtGui import QFont, QPixmap, QIcon, QColor, QPalette, QLinearGradient
import torch
import time
import cv2
import numpy as np

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
            for i in range(30):
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
                
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False


class KalmanFilter:
    def __init__(self, state_dim, measurement_dim, process_noise_cov=1e-2, measurement_noise_cov=1e-1, error_cov_post=1.0):
        self.state_dim = state_dim # Store these for later access if needed
        self.measurement_dim = measurement_dim
        self.kf = cv2.KalmanFilter(state_dim, measurement_dim)
        
        # State transition matrix (A)
        # For a constant velocity model with (x, y, dx, dy)
        # If state is (x, y, w, h, vx, vy, vw, vh), then state_dim = 8
        # If measurement is (x, y, w, h), then measurement_dim = 4
        # We need to define A, H, Q, R, P

        # Initialize with reasonable default matrices for common use cases.
        # These will need to be adjusted based on your specific state and measurement definitions.

        # Example: Constant velocity model for (x, y, dx, dy)
        # State: [x, y, vx, vy]
        # Measurement: [x, y]
        dt = 1.0 # Time step (assuming 1 frame per step)

        if state_dim == 4 and measurement_dim == 2: # For (x, y, vx, vy) state and (x, y) measurement
            self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                                 [0, 1, 0, dt],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
            self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        elif state_dim == 8 and measurement_dim == 4: # For (x1, y1, x2, y2, vx1, vy1, vx2, vy2) state and (x1, y1, x2, y2) measurement
            self.kf.transitionMatrix = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                                                 [0, 1, 0, 0, 0, dt, 0, 0],
                                                 [0, 0, 1, 0, 0, 0, dt, 0],
                                                 [0, 0, 0, 1, 0, 0, 0, dt],
                                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
            self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)
        else:
            raise ValueError("Unsupported state and measurement dimensions for KalmanFilter initialization.")


        self.kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise_cov # Q
        self.kf.measurementNoiseCov = np.eye(measurement_dim, dtype=np.float32) * measurement_noise_cov # R
        self.kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * error_cov_post # P

    def predict(self):
        return self.kf.predict()

    def update(self, measurement):
        # return self.kf.correct(measurement)
        # Ensure the measurement is correctly shaped and typed
        if measurement.shape != (self.measurement_dim, 1): # kf.measureSize is the correct way here
            raise ValueError(f"Measurement has incorrect shape: {measurement.shape}. Expected ({self.kf.measureSize}, 1).")
        if measurement.dtype != np.float32:
            measurement = measurement.astype(np.float32)

        # --- ADD THESE DEBUG PRINTS ---
        print(f"DEBUG KF Update: kf.statePost BEFORE correct: {self.kf.statePost.flatten()}")
        print(f"DEBUG KF Update: kf.errorCovPost BEFORE correct: {np.diag(self.kf.errorCovPost)}")
        print(f"DEBUG KF Update: Measurement received: {measurement.flatten()}")
        # --- END DEBUG PRINTS ---

        corrected_state = self.kf.correct(measurement)

        # --- ADD THESE DEBUG PRINTS ---
        print(f"DEBUG KF Update: kf.statePost AFTER correct: {self.kf.statePost.flatten()}")
        print(f"DEBUG KF Update: kf.errorCovPost AFTER correct: {np.diag(self.kf.errorCovPost)}")
        # --- END DEBUG PRINTS ---

        return corrected_state



"""

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim, process_noise_cov=1e-2, measurement_noise_cov=1e-1, error_cov_post=1.0):
        self.kf = cv2.KalmanFilter(state_dim, measurement_dim)
        # Store these for later use in the update method
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim 
        
        dt = 1.0 # Time step (assuming 1 frame per step)

        # Transition Matrix (A)
        # Defines how the state evolves from t to t+1
        if state_dim == 8:
            self.kf.transitionMatrix = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                                                 [0, 1, 0, 0, 0, dt, 0, 0],
                                                 [0, 0, 1, 0, 0, 0, dt, 0],
                                                 [0, 0, 0, 1, 0, 0, 0, dt],
                                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        elif state_dim == 4: # For the 4-state test
            self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                                 [0, 1, 0, dt],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        else:
            raise ValueError("Unsupported state_dim")

        # Measurement Matrix (H)
        # Defines how the state maps to the measurement
        if measurement_dim == 4 and state_dim == 8:
            self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)
        elif measurement_dim == 2 and state_dim == 4: # For the 4-state test
            self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        else:
            raise ValueError("Unsupported measurement_dim or state_dim combination")

        # Process Noise Covariance (Q)
        # Uncertainty in the state transition model
        self.kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise_cov

        # Measurement Noise Covariance (R)
        # Uncertainty in the measurement
        self.kf.measurementNoiseCov = np.eye(measurement_dim, dtype=np.float32) * measurement_noise_cov
        
        # Error Covariance (P) - initialized after correction
        # Initial estimate of the state error covariance
        self.kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * error_cov_post 

        # Give higher initial uncertainty to position components
        if state_dim == 8:
            for i in range(4): # x1, y1, x2, y2
                self.kf.errorCovPost[i,i] = 10000.0 # High uncertainty
            for i in range(4, 8): # vx1, vy1, vx2, vy2
                self.kf.errorCovPost[i,i] = 100.0 # Lower uncertainty
        elif state_dim == 4: # For the 4-state test: x, y, vx, vy
            # Set all position and velocity states to high uncertainty initially
            self.kf.errorCovPost[0,0] = 10000.0 # x
            self.kf.errorCovPost[1,1] = 10000.0 # y
            self.kf.errorCovPost[2,2] = 100.0 # vx
            self.kf.errorCovPost[3,3] = 100.0 # vy


    def predict(self):
        return self.kf.predict()

    def update(self, measurement):
        # Ensure the measurement is correctly shaped and typed
        # The measurement must be a column vector of shape (self.measurement_dim, 1) and type CV_32F
        # Use self.measurement_dim directly, as kf.measureSize is not exposed in Python
        if measurement.shape != (self.measurement_dim, 1): # <--- FIXED LINE HERE
            raise ValueError(f"Measurement has incorrect shape: {measurement.shape}. Expected ({self.measurement_dim}, 1).")
        if measurement.dtype != np.float32:
            measurement = measurement.astype(np.float32)

        # --- ADD THESE DEBUG PRINTS (if not already there) ---
        print(f"DEBUG KF Update: kf.statePost BEFORE correct: {self.kf.statePost.flatten()}")
        print(f"DEBUG KF Update: kf.errorCovPost BEFORE correct: {np.diag(self.kf.errorCovPost)}")
        print(f"DEBUG KF Update: Measurement received: {measurement.flatten()}")
        # --- END DEBUG PRINTS ---

        corrected_state = self.kf.correct(measurement)
        
        # --- ADD THESE DEBUG PRINTS (if not already there) ---
        print(f"DEBUG KF Update: kf.statePost AFTER correct: {self.kf.statePost.flatten()}")
        print(f"DEBUG KF Update: kf.errorCovPost AFTER correct: {np.diag(self.kf.errorCovPost)}")
        # --- END DEBUG PRINTS ---

        return corrected_state
    """