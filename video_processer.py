import numpy as np
import cv2
import torch
from PyQt5.QtCore import (QRunnable, pyqtSlot, pyqtSignal, QObject)
from PyQt5.QtWidgets import QApplication
import sys

class FrameProcessorSignals(QObject):
    finished = pyqtSignal(np.ndarray, float, int)
    error = pyqtSignal(str)

class FrameProcessor(QRunnable):
    def __init__(self, frame, model, threshold, device, img_size=640):
        super().__init__()
        self.frame = frame
        self.model = model
        self.threshold = threshold
        self.device = device
        self.img_size = img_size
        self.signals = FrameProcessorSignals()
        self.previous_detections = []
        self.previous_frame_time = None

    @pyqtSlot()
    def run(self):
        try:
            processed_frame, confidence, bbox_count = self.process_frame_with_model()
            self.signals.finished.emit(processed_frame, confidence, bbox_count)
        except Exception as e:
            self.signals.error.emit(str(e))

    def process_frame_with_model(self):
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        resized_frame = self._letterbox(frame_rgb, new_shape=(self.img_size, self.img_size))
        tensor = torch.from_numpy(resized_frame).float().to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            results = self.model(tensor)

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        keep = scores > (self.threshold / 100)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        original_frame_height, original_frame_width = self.frame.shape[:2]
        boxes = self._scale_boxes(boxes, resized_frame.shape[:2], (original_frame_height, original_frame_width))

        display_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

        detected_lines_dict = self._detect_court_lines(display_frame)

        for line_name, line_coords in detected_lines_dict.items():
            if line_coords is not None:
                x1, y1, x2, y2 = line_coords
                if line_name == 'vertical_service_line':
                    cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 3) # Yellow, thicker

        processed_frame = self._draw_detections(display_frame, boxes, scores, class_ids)

        avg_confidence = np.mean(scores) * 100 if len(scores) > 0 else 0
        bbox_count = len(boxes)

        # cv2.destroyAllWindows() # Close all debug windows before returning

        return processed_frame, avg_confidence, bbox_count

    def _scale_boxes(self, boxes, from_shape, to_shape):
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad_x = (from_shape[1] - to_shape[1] * gain) / 2
        pad_y = (from_shape[0] - to_shape[0] * gain) / 2

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, :4] /= gain

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, to_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, to_shape[0])

        return boxes

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return img

    def _draw_detections(self, frame, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            color = self._get_color(class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{self.model.names[class_id]} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return frame

    def _get_color(self, class_id):
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0),
        ]
        return colors[class_id % len(colors)]

    def _detect_court_lines(self, frame):
        detected_lines_dict = {}
        height, width = frame.shape[:2]

        # Assuming Canny and ROI are already well-tuned.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3) # Adjust if needed for better edge clarity.

        # Example ROI that matches image_f24afa.png roughly
        # YOU MUST ADJUST THESE TO PERFECTLY MATCH YOUR EXACT LINE
        vertical_service_line_roi = np.array([
            [int(width * 0.49), int(height * 0.552)],  # Start from top of frame
            [int(width * 0.507), int(height * 0.99)]   # Go almost to bottom
        ], dtype=np.int32)
        # Note: If the line starts lower, adjust height * 0.0 upwards (e.g., height * 0.40)
        # If the line ends higher, adjust height * 0.99 downwards (e.g., height * 0.80)
        # Adjust width * 0.61 and width * 0.64 to precisely capture the line horizontally.

        vertical_roi_polygon = np.array([
            [vertical_service_line_roi[0,0], vertical_service_line_roi[0,1]],
            [vertical_service_line_roi[1,0], vertical_service_line_roi[0,1]],
            [vertical_service_line_roi[1,0], vertical_service_line_roi[1,1]],
            [vertical_service_line_roi[0,0], vertical_service_line_roi[1,1]]
        ], dtype=np.int32)

        mask_vertical = np.zeros_like(edges)
        cv2.fillPoly(mask_vertical, [vertical_roi_polygon], 255)
        masked_edges_vertical = cv2.bitwise_and(edges, mask_vertical)

        # This should now show your desired vertical line clearly and isolated.
        #cv2.imshow("Masked Edges Vertical (Debugging)", masked_edges_vertical)
        #cv2.waitKey(0)

        # --- TUNING HOUGH LINES P PARAMETERS for image_f24afa.png ---
        # Goal: Detect a single, continuous line, even with occlusion.
        lines_vertical = cv2.HoughLinesP(
            masked_edges_vertical,
            rho=1,
            theta=np.pi / 180,
            threshold=15,           # Lowered threshold to pick up more segments, even weak ones
            minLineLength=int(height * 0.35), # Slightly increased to favor longer lines. Adjust as needed.
            maxLineGap=80           # SIGNIFICANTLY increased to bridge the occlusion gaps
        )

        best_vertical_line = None
        if lines_vertical is not None:
            max_line_length = 0
            for line in lines_vertical:
                x1, y1, x2, y2 = line[0]

                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)

                if angle_deg < 0:
                    angle_deg += 180

                # Angle range: very robust for slight perspective variations
                # The line in image_f24afa.png seems quite straight.
                # If it's slightly angled, this range is good.
                if 70 <= angle_deg <= 110: # (90 +/- 20)
                    current_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                    # Prioritize the longest valid line
                    if current_length > max_line_length:
                        max_line_length = current_length
                        best_vertical_line = (x1, y1, x2, y2)
        
        # Post-processing: If best_vertical_line is found, "smooth" it or extend it.
        # This will make the line appear continuous despite occlusions.
        if best_vertical_line is not None:
            x1, y1, x2, y2 = best_vertical_line
            
            # Calculate slope (m) and y-intercept (b)
            if (x2 - x1) != 0:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                
                # Extend to the top and bottom Y-coordinates of your ROI
                # This makes the line appear "solid" and reach the ROI boundaries
                y_top_extended = vertical_service_line_roi[0,1]
                x_top_extended = int((y_top_extended - b) / m) if m != 0 else x1

                y_bottom_extended = vertical_service_line_roi[1,1]
                x_bottom_extended = int((y_bottom_extended - b) / m) if m != 0 else x2
                
                best_vertical_line = (x_top_extended, y_top_extended, x_bottom_extended, y_bottom_extended)
            else: # Perfectly vertical line (x1 == x2)
                best_vertical_line = (x1, vertical_service_line_roi[0,1], x1, vertical_service_line_roi[1,1])


        detected_lines_dict['vertical_service_line'] = best_vertical_line

        return detected_lines_dict