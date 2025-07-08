from PyQt5.QtCore import (QRunnable, QThreadPool, pyqtSlot, 
                         pyqtSignal, QObject)
import numpy as np
import cv2
import torch

class FrameProcessorSignals(QObject):
    finished = pyqtSignal(np.ndarray, float, int)  # processed_frame, confidence, bbox_count
    error = pyqtSignal(str)

class FrameProcessor(QRunnable):
    def __init__(self, frame, model, threshold, device, img_size=640):
        super().__init__()
        self.frame = frame
        self.model = model
        self.threshold = threshold
        self.device = device  # Add device parameter
        self.img_size = img_size  # Target size for model (must be divisible by 32)
        # self.original_size = original_size or frame.shape[:2]
        self.signals = FrameProcessorSignals()
    
    @pyqtSlot()
    def run(self):
        try:
            # Process frame with your model
            processed_frame, confidence, bbox_count = self.process_frame_with_model()
            self.signals.finished.emit(processed_frame, confidence, bbox_count)
        except Exception as e:
            self.signals.error.emit(str(e))
    
    def process_frame_with_model(self):
        # Convert BGR to RGB and resize if needed
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to model's expected size while maintaining aspect ratio
        resized_frame = self._letterbox(frame_rgb, new_shape=(self.img_size, self.img_size))

        # Convert to tensor and move to device
        tensor = torch.from_numpy(resized_frame).float().to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # HWC to BCHW and normalize
        
        # Run inference
        with torch.no_grad():
            results = self.model(tensor)
        
        # Process results
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Filter by threshold
        keep = scores > (self.threshold / 100)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # SCALE BOXES BACK TO ORIGINAL FRAME SIZE HERE
        boxes = self._scale_boxes(boxes, resized_frame.shape[:2], self.frame.shape[:2])
        
        # Convert back to BGR for display
        display_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

        # Draw detections on original frame (not the RGB one)
        processed_frame = self._draw_detections(display_frame, boxes, scores, class_ids)
        avg_confidence = np.mean(scores) * 100 if len(scores) > 0 else 0
        bbox_count = len(boxes)
        
        return processed_frame, avg_confidence, bbox_count
    
    def _scale_boxes(self, boxes, from_shape, to_shape):
        """Rescale bounding boxes from resized frame to original frame"""
        # Calculate scaling factors
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad = (from_shape[1] - to_shape[1] * gain) / 2, (from_shape[0] - to_shape[0] * gain) / 2
        
        # Scale boxes
        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
        
        # Clip boxes to image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, to_shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, to_shape[0])  # y1, y2
        
        return boxes


    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize image with aspect ratio unchanged using letterbox method"""
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=color)
        return img


    def _draw_detections(self, frame, boxes, scores, class_ids):
        """Draw bounding boxes and labels on the frame"""
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Green

            # Generate a unique color for each class ID
            # You can use a predefined color palette or generate colors dynamically
            color = self._get_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{self.model.names[class_id]} {score:.2f}"
            
            # Calculate text size
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            
            # Put text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return frame
    
    def _get_color(self, class_id):
        """Generate a unique color for each class ID"""
        # You can use different approaches:
        
        # 1. Predefined color palette (add more colors as needed)
        colors = [
            (0, 255, 0),    # green
            (255, 0, 0),    # blue
            (0, 0, 255),    # red
            # (255, 255, 0),  # cyan
            # (0, 255, 255),  # yellow
            # (255, 0, 255), # magenta
            # (128, 0, 0),
            # (0, 128, 0),
            # (0, 0, 128),
            # (128, 128, 0),
        ]
        
        # Return a color based on class_id (cycling through the palette if needed)
        return colors[class_id % len(colors)]