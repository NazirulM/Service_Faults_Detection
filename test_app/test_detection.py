import numpy as np
import cv2
import torch
from PyQt5.QtCore import (QRunnable, pyqtSlot, pyqtSignal, QObject, QTimer, Qt)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QPushButton, QFileDialog, QSlider, QHBoxLayout,
                             QMessageBox, QLineEdit) # Added QLineEdit for model path input
from PyQt5.QtGui import QImage, QPixmap
import sys

# Ensure utils/utility.py is accessible and contains your KalmanFilter
# Assuming your KalmanFilter from the original post is in utils/utility.py
from utils.utility import KalmanFilter

# Your TrackedObject class (from original code, unchanged)
class TrackedObject:
    """Represents a single tracked object with its own Kalman filter."""
    def __init__(self, bbox, class_id, score, obj_id):
        self.id = obj_id
        # State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.kf = KalmanFilter(state_dim=8, measurement_dim=4,
                                process_noise_cov=1e-2, measurement_noise_cov=0.1, error_cov_post=1.0)
        
        # Initialize Kalman filter state with current bbox
        initial_state = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0], np.float32).reshape(-1, 1)
        self.kf.kf.statePost = initial_state
        
        self.bbox = bbox
        self.class_id = class_id
        self.score = score
        self.frames_since_last_update = 0
        self.active = True

    def update(self, new_bbox, new_score):
        measurement = np.array(new_bbox, np.float32).reshape(-1, 1)
        predicted_state = self.kf.predict() # Predict before update for consistency
        corrected_state = self.kf.update(measurement)
        
        # Update internal bbox with corrected state
        self.bbox = corrected_state[:4].flatten()
        self.score = new_score
        self.frames_since_last_update = 0
        return self.bbox

    def predict(self):
        predicted_state = self.kf.predict()
        self.bbox = predicted_state[:4].flatten()
        self.frames_since_last_update += 1
        return self.bbox

# Your FrameProcessor Class (re-integrated with no QRunnable/QObject for direct calls)
class FrameProcessor:
    def __init__(self, model, threshold, device, img_size=640):
        self.frame = None
        self.model = model # This will be the actual loaded YOLO model
        self.threshold = threshold
        self.device = device
        self.img_size = img_size
        self.previous_detections = []
        self.previous_frame_time = None
        self.frame_count = 0
        
        self.tracked_objects = []
        self.next_object_id = 0
        self.max_frames_to_keep_track = 10

        self.vertical_service_line_kf = None
        self.line_kf_process_noise = 1e-3
        self.line_kf_measurement_noise = 1e-1
        self.line_kf_error_cov_post = 1.0

        self.last_known_vertical_line = None
        self.frames_without_line_detection = 0
        self.max_frames_to_persist_line = 5

    def set_current_frame(self, frame):
        self.frame = frame

    def process_frame_with_model(self):
        if self.frame is None:
            print("Error: No frame to process.")
            return None, 0, 0

        if self.model is None:
            print("Error: YOLO Model not loaded.")
            return self.frame, 0, 0 # Return original frame if model isn't loaded yet

        self.frame_count += 1

        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        resized_frame = self._letterbox(frame_rgb, new_shape=(self.img_size, self.img_size))
        tensor = torch.from_numpy(resized_frame).float().to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

        # --- Real Model Inference ---
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
        scaled_boxes = self._scale_boxes(boxes, resized_frame.shape[:2], (original_frame_height, original_frame_width))

        display_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

        # --- Kalman Filter for Object Detections ---
        for track in self.tracked_objects:
            if track.active:
                track.predict()

        new_detections = []
        for i in range(len(scaled_boxes)):
            new_detections.append({
                'bbox': scaled_boxes[i],
                'score': scores[i],
                'class_id': class_ids[i]
            })

        self._associate_detections_to_tracks(new_detections)

        self.tracked_objects = [track for track in self.tracked_objects if track.active and track.frames_since_last_update <= self.max_frames_to_keep_track]
        
        tracked_boxes = []
        tracked_scores = []
        tracked_class_ids = []
        
        for track in self.tracked_objects:
            if track.active:
                tracked_boxes.append(track.bbox)
                tracked_scores.append(track.score)
                tracked_class_ids.append(track.class_id)

        tracked_boxes = np.array(tracked_boxes)
        tracked_scores = np.array(tracked_scores)
        tracked_class_ids = np.array(tracked_class_ids)

        # --- Kalman Filter for Line Detection ---
        detected_lines_dict = self._detect_court_lines(display_frame)
        
        current_vertical_line = detected_lines_dict.get('vertical_service_line')

        if current_vertical_line is not None:
            if self.vertical_service_line_kf is None:
                self.vertical_service_line_kf = KalmanFilter(
                    state_dim=8, measurement_dim=4,
                    process_noise_cov=self.line_kf_process_noise,
                    measurement_noise_cov=self.line_kf_measurement_noise,
                    error_cov_post=self.line_kf_error_cov_post
                )
                initial_line_state = np.array([current_vertical_line[0], current_vertical_line[1],
                                               current_vertical_line[2], current_vertical_line[3],
                                               0, 0, 0, 0], np.float32).reshape(-1, 1)
                self.vertical_service_line_kf.kf.statePost = initial_line_state
            
            measurement_line = np.array(current_vertical_line, np.float32).reshape(-1, 1)
            corrected_line_state = self.vertical_service_line_kf.update(measurement_line)
            
            self.last_known_vertical_line = corrected_line_state[:4].flatten().astype(int)
            self.frames_without_line_detection = 0
        else:
            if self.vertical_service_line_kf is not None:
                predicted_line_state = self.vertical_service_line_kf.predict()
                self.last_known_vertical_line = predicted_line_state[:4].flatten().astype(int)
                self.frames_without_line_detection += 1
                
                if self.frames_without_line_detection > self.max_frames_to_persist_line:
                    self.last_known_vertical_line = None

        if self.last_known_vertical_line is not None:
            x1_line, y1_line, x2_line, y2_line = self.last_known_vertical_line
            cv2.line(display_frame, (x1_line, y1_line), (x2_line, y2_line), (0, 255, 255), 3)
        
        processed_frame = self._draw_detections(display_frame, tracked_boxes, tracked_scores, tracked_class_ids)

        avg_confidence = np.mean(tracked_scores) * 100 if len(tracked_scores) > 0 else 0
        bbox_count = len(tracked_boxes)

        return processed_frame, avg_confidence, bbox_count

    def _associate_detections_to_tracks(self, detections):
        if not self.tracked_objects:
            for det in detections:
                self.tracked_objects.append(TrackedObject(det['bbox'], det['class_id'], det['score'], self.next_object_id))
                self.next_object_id += 1
            return

        for track in self.tracked_objects:
            track.active = False

        unassigned_detections_indices = list(range(len(detections)))

        for i, track in enumerate(self.tracked_objects):
            if not track.active:
                best_match_idx = -1
                min_distance = float('inf')

                for j in unassigned_detections_indices:
                    det = detections[j]
                    center_track = ((track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2)
                    center_det = ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2)
                    
                    dist = np.linalg.norm(np.array(center_track) - np.array(center_det))

                    if dist < min_distance and dist < 100:
                        min_distance = dist
                        best_match_idx = j
                
                if best_match_idx != -1:
                    track.update(detections[best_match_idx]['bbox'], detections[best_match_idx]['score'])
                    track.active = True
                    unassigned_detections_indices.remove(best_match_idx)
            
        for idx in unassigned_detections_indices:
            det = detections[idx]
            self.tracked_objects.append(TrackedObject(det['bbox'], det['class_id'], det['score'], self.next_object_id))
            self.next_object_id += 1
    
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
        if self.model is None or not hasattr(self.model, 'names'):
            class_names = {i: f"Class {i}" for i in range(10)} # Fallback dummy names
        else:
            class_names = self.model.names

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            color = self._get_color(class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names.get(class_id, 'Unknown')} {score:.2f}"
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # ROI for vertical service line (ADJUST THESE VALUES FOR YOUR VIDEO/IMAGE)
        # These are example values. You will need to fine-tune them based on your camera angle and court layout.
        # It's helpful to display `edges` and draw a rectangle on it to find good coordinates.
        vertical_service_line_roi_x_min = int(width * 0.49)
        vertical_service_line_roi_x_max = int(width * 0.507)
        vertical_service_line_roi_y_min = int(height * 0.552)
        vertical_service_line_roi_y_max = int(height * 0.99)

        vertical_roi_polygon = np.array([
            [vertical_service_line_roi_x_min, vertical_service_line_roi_y_min],
            [vertical_service_line_roi_x_max, vertical_service_line_roi_y_min],
            [vertical_service_line_roi_x_max, vertical_service_line_roi_y_max],
            [vertical_service_line_roi_x_min, vertical_service_line_roi_y_max]
        ], dtype=np.int32)

        mask_vertical = np.zeros_like(edges)
        cv2.fillPoly(mask_vertical, [vertical_roi_polygon], 255)
        masked_edges_vertical = cv2.bitwise_and(edges, mask_vertical)

        # Debugging: Show the masked edges
        #cv2.imshow("Masked Edges Vertical (Debugging)", masked_edges_vertical)
        # cv2.waitKey(0) # Uncomment to pause and inspect each frame's masked edges

        lines_vertical = cv2.HoughLinesP(
            masked_edges_vertical,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=int(height * 0.25),
            maxLineGap=80
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

                # Filter for lines that are close to vertical
                if 70 <= angle_deg <= 110:
                    current_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                    if current_length > max_line_length:
                        max_line_length = current_length
                        best_vertical_line = (x1, y1, x2, y2)
        
        if best_vertical_line is not None:
            x1, y1, x2, y2 = best_vertical_line
            
            y_roi_top = vertical_service_line_roi_y_min
            y_roi_bottom = vertical_service_line_roi_y_max
            
            if (x2 - x1) != 0:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                
                x_top_extended = int(np.clip((y_roi_top - b) / m, 0, width - 1)) if m != 0 else x1
                x_bottom_extended = int(np.clip((y_roi_bottom - b) / m, 0, width - 1)) if m != 0 else x2
                
                best_vertical_line = (x_top_extended, y_roi_top, x_bottom_extended, y_roi_bottom)
            else:
                best_vertical_line = (x1, y_roi_top, x1, y_roi_bottom)

        detected_lines_dict['vertical_service_line'] = best_vertical_line

        return detected_lines_dict

# --- Main Debugging Application ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Court Line & Object Tracking Debugger (Real Model)")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Model loading section
        self.model_layout = QHBoxLayout()
        self.model_path_label = QLabel("YOLO Model Path:")
        self.model_layout.addWidget(self.model_path_label)
        self.model_path_input = QLineEdit("best.pt") # Default path
        self.model_layout.addWidget(self.model_path_input)
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.clicked.connect(self.browse_model_path)
        self.model_layout.addWidget(self.browse_model_btn)
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_yolo_model)
        self.model_layout.addWidget(self.load_model_btn)
        self.layout.addLayout(self.model_layout)

        self.model = None # Actual YOLO model will be loaded here
        self.frame_processor = None # Will be initialized after model loads

        self.image_label = QLabel("Load a video or image to start processing.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Controls Layout
        self.controls_layout = QHBoxLayout()
        
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        self.controls_layout.addWidget(self.load_video_btn)

        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.controls_layout.addWidget(self.load_image_btn)

        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setEnabled(False)
        self.controls_layout.addWidget(self.play_pause_btn)

        self.next_frame_btn = QPushButton("Next Frame")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        self.controls_layout.addWidget(self.next_frame_btn)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.sliderMoved.connect(self.set_frame_from_slider)
        self.frame_slider.setEnabled(False)
        self.controls_layout.addWidget(self.frame_slider)

        self.layout.addLayout(self.controls_layout)

        self.status_label = QLabel("Status: Ready. Please load YOLO model.")
        self.layout.addWidget(self.status_label)

        self.cap = None
        self.total_frames = 0
        self.current_frame_pos = 0
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(33)

        # Create a named window for debugging masked edges (optional, but good practice)
        cv2.namedWindow("Masked Edges Vertical (Debugging)", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        # cv2.setWindowProperty("Masked Edges Vertical (Debugging)", cv2.WND_PROP_TOPMOST, 1) # This can be annoying

        self._set_media_controls_enabled(False) # Disable media controls until model is loaded

    def _set_media_controls_enabled(self, enabled):
        self.load_video_btn.setEnabled(enabled)
        self.load_image_btn.setEnabled(enabled)
        self.play_pause_btn.setEnabled(enabled and self.cap is not None)
        self.next_frame_btn.setEnabled(enabled and self.cap is not None)
        self.frame_slider.setEnabled(enabled and self.cap is not None)

    def browse_model_path(self):
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(self, "Select YOLO Model", "", "Model Files (*.pt *.pth);;All Files (*)")
        if model_path:
            self.model_path_input.setText(model_path)

    def load_yolo_model(self):
        model_path = self.model_path_input.text()
        if not model_path:
            QMessageBox.warning(self, "Error", "Please provide a YOLO model path.")
            return

        try:
            # Use torch.hub.load or your direct model loading method
            # For local files, it's often better to use:
            # from ultralytics import YOLO
            # self.model = YOLO(model_path)
            
            # This line assumes you have 'ultralytics' installed and can directly load
            # from a local path using the YOLO constructor.
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            
            # Optionally set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device) # Move model to device

            # Initialize FrameProcessor with the real model
            self.frame_processor = FrameProcessor(
                model=self.model,
                threshold=25, # Default threshold, can be made configurable
                device=device,
                img_size=640 # Default image size, can be made configurable
            )
            self.status_label.setText(f"YOLO Model loaded successfully from {model_path} on {device}.")
            self._set_media_controls_enabled(True) # Enable media controls
            print(f"YOLO Model loaded successfully from {model_path} on {device}.")
            print(f"Model Class Names: {self.model.names}")

        except Exception as e:
            self.model = None
            self.frame_processor = None
            self._set_media_controls_enabled(False)
            QMessageBox.critical(self, "Model Load Error", f"Failed to load YOLO model: {e}")
            self.status_label.setText("Status: Model loading failed.")
            print(f"Failed to load YOLO model: {e}")

    def load_video(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a YOLO model first.")
            return

        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.status_label.setText(f"Error: Could not open video {video_path}")
                self.reset_state()
                return

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_pos = 0
            self.frame_slider.setMaximum(self.total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.play_pause_btn.setEnabled(True)
            self.next_frame_btn.setEnabled(True)
            self.status_label.setText(f"Loaded video: {video_path} ({self.total_frames} frames)")
            self.playing = False
            self.play_pause_btn.setText("Play")
            self.read_and_process_frame(0)
            self.frame_processor.tracked_objects = [] # Reset tracks for new video
            self.frame_processor.vertical_service_line_kf = None # Reset line KF for new video

    def load_image(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a YOLO model first.")
            return

        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if image_path:
            self.cap = None
            self.total_frames = 1
            self.current_frame_pos = 0
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(False)
            self.play_pause_btn.setEnabled(False)
            self.next_frame_btn.setEnabled(False)
            self.status_label.setText(f"Loaded image: {image_path}")
            
            frame = cv2.imread(image_path)
            if frame is None:
                self.status_label.setText(f"Error: Could not load image {image_path}")
                self.reset_state()
                return
            
            self.frame_processor.set_current_frame(frame)
            processed_frame, confidence, bbox_count = self.frame_processor.process_frame_with_model()
            self.display_frame(processed_frame)
            self.status_label.setText(f"Processed image. Detections: {bbox_count}")
            self.frame_processor.tracked_objects = [] # Reset tracks for new image
            self.frame_processor.vertical_service_line_kf = None # Reset line KF for new image


    def read_and_process_frame(self, frame_number):
        if self.cap is None:
            return

        if frame_number >= self.total_frames or frame_number < 0:
            if self.playing:
                self.toggle_play_pause()
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText(f"Error: Could not read frame {frame_number}")
            self.toggle_play_pause()
            return

        self.current_frame_pos = frame_number
        self.frame_slider.setValue(frame_number)

        if self.frame_processor:
            self.frame_processor.set_current_frame(frame)
            processed_frame, confidence, bbox_count = self.frame_processor.process_frame_with_model()
            self.display_frame(processed_frame)
            self.status_label.setText(f"Frame {frame_number}/{self.total_frames-1} | Detections: {bbox_count} | Confidence: {confidence:.2f}%")
        else:
            self.display_frame(frame) # Display raw frame if processor not ready
            self.status_label.setText(f"Frame {frame_number}/{self.total_frames-1} | Model not loaded.")
        
    def display_frame(self, frame):
        if frame is None:
            return
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)

        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def toggle_play_pause(self):
        self.playing = not self.playing
        if self.playing:
            self.play_pause_btn.setText("Pause")
            self.timer.start()
        else:
            self.play_pause_btn.setText("Play")
            self.timer.stop()

    def next_frame(self):
        if self.cap is not None:
            self.read_and_process_frame(self.current_frame_pos + 1)
        
    def set_frame_from_slider(self, position):
        if self.cap is not None:
            self.read_and_process_frame(position)

    def reset_state(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self.total_frames = 0
        self.current_frame_pos = 0
        self.playing = False
        self.timer.stop()
        self.image_label.clear()
        self.image_label.setText("Load a video or image to start processing.")
        self.play_pause_btn.setText("Play")
        self.play_pause_btn.setEnabled(False)
        self.next_frame_btn.setEnabled(False)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.status_label.setText("Status: Ready. Please load YOLO model.")
        if self.frame_processor:
            self.frame_processor.tracked_objects = []
            self.frame_processor.vertical_service_line_kf = None
        cv2.destroyAllWindows()

    def closeEvent(self, event):
        self.reset_state()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())