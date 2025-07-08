import cv2
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, pyqtSlot
import numpy as np

class FrameProcessor(QRunnable):
    def __init__(self, frame, model):
        super().__init__()
        self.frame = frame
        self.model = model
        self.signals = ProcessorSignals()
        
    @pyqtSlot()
    def run(self):
        try:
            # Preprocess frame
            processed_frame = self.preprocess(self.frame)
            
            # Run model inference
            results = self.model.predict(processed_frame)
            
            # Emit results
            self.signals.result.emit((self.frame, results))
            
        except Exception as e:
            self.signals.error.emit(str(e))
    
    def preprocess(self, frame):
        """Convert frame to model input format"""
        # Example: Resize, normalize, etc.
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        return np.expand_dims(frame, axis=0)

class ProcessorSignals(QObject):
    result = pyqtSignal(tuple)  # (original_frame, processing_results)
    error = pyqtSignal(str)