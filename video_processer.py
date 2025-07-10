import numpy as np
import cv2
import torch
from PyQt5.QtCore import (QRunnable, pyqtSlot, pyqtSignal, QObject)
from PyQt5.QtWidgets import QApplication
import sys
from utils.utility import KalmanFilter

class FrameProcessorSignals(QObject):
    finished = pyqtSignal(np.ndarray, float, int, list, int, object, tuple, int)
    error = pyqtSignal(str)

class TrackedObject:
    """Represents a single tracked object with its own Kalman filter."""
    def __init__(self, bbox, class_id, score, obj_id):
        self.id = obj_id
        # State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.kf = KalmanFilter(state_dim=8, measurement_dim=4,
                               process_noise_cov=1e-2, measurement_noise_cov=0.1, error_cov_post=1000.0)
        
        # Initialize Kalman filter state with current bbox
        initial_state = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0], np.float32).reshape(-1, 1)
        self.kf.kf.statePost = initial_state

        # DEBUG: Print initial state of each new TrackedObject KF
        print(f"TrackedObject {self.id}: KF Initialized. statePost: {self.kf.kf.statePost.flatten()}")
        print(f"TrackedObject {self.id}: KF Initialized. errorCovPost diag: {np.diag(self.kf.kf.errorCovPost)}")
        
        self.bbox = bbox
        self.class_id = class_id
        self.score = score
        #self.frames_since_last_update = 0
        self.active = True

        self.frames_since_last_update = 0 # Counter for consecutive frames without an update

    def update(self, new_bbox, new_score):
        measurement = np.array(new_bbox, np.float32).reshape(-1, 1)
        #predicted_state = self.kf.predict() # Predict before update for consistency
        
        # DEBUG: Print BEFORE update
        print(f"TrackedObject {self.id} (UPDATE): Measurement: {measurement.flatten()}")
        print(f"TrackedObject {self.id} (UPDATE): kf.statePost BEFORE correct: {self.kf.kf.statePost.flatten()}")
        print(f"TrackedObject {self.id} (UPDATE): kf.errorCovPost BEFORE correct: {np.diag(self.kf.kf.errorCovPost)}")

        corrected_state = self.kf.update(measurement)
        
        # DEBUG: Print AFTER update
        print(f"TrackedObject {self.id} (UPDATE): kf.statePost AFTER correct: {self.kf.kf.statePost.flatten()}")
        print(f"TrackedObject {self.id} (UPDATE): kf.errorCovPost AFTER correct: {np.diag(self.kf.kf.errorCovPost)}")
        print(f"TrackedObject {self.id} (UPDATE): Corrected state returned: {corrected_state.flatten()}")

        # Update internal bbox with corrected state
        self.bbox = corrected_state[:4].flatten()
        self.score = new_score
        self.frames_since_last_update = 0
        return self.bbox

    def predict(self):
        # DEBUG: Print BEFORE predict
        print(f"TrackedObject {self.id} (PREDICT): kf.statePost BEFORE predict: {self.kf.kf.statePost.flatten()}")
        print(f"TrackedObject {self.id} (PREDICT): kf.errorCovPost BEFORE predict: {np.diag(self.kf.kf.errorCovPost)}")

        predicted_state = self.kf.predict()

        # DEBUG: Print AFTER predict
        print(f"TrackedObject {self.id} (PREDICT): kf.statePost AFTER predict: {self.kf.kf.statePost.flatten()}")
        print(f"TrackedObject {self.id} (PREDICT): kf.errorCovPost AFTER predict: {np.diag(self.kf.kf.errorCovPost)}")
        print(f"TrackedObject {self.id} (PREDICT): Predicted state returned: {predicted_state.flatten()}")

        self.bbox = predicted_state[:4].flatten()
        self.frames_since_last_update += 1
        return self.bbox

class FrameProcessor(QRunnable):
    def __init__(self, frame, model, threshold, device, img_size=640,
                 current_tracked_objects=None, current_next_object_id=0,
                 current_vertical_service_line_kf=None, current_last_known_vertical_line=None,
                 current_frames_without_line_detection=0):
        super().__init__()
        self.frame = frame
        self.model = model
        self.threshold = threshold
        self.device = device
        self.img_size = img_size
        self.signals = FrameProcessorSignals()
        self.previous_detections = []
        self.previous_frame_time = None
        
        # For object tracking with Kalman Filter
        self.tracked_objects = current_tracked_objects if current_tracked_objects is not None else []
        self.next_object_id = current_next_object_id
        # self.max_frames_to_keep_track = 10 # How many frames to keep a track without detection

         # Kalman Filter for the vertical service line
        self.vertical_service_line_kf = current_vertical_service_line_kf
        self.last_known_vertical_line = current_last_known_vertical_line
        self.frames_without_line_detection = current_frames_without_line_detection
        # State for vertical line: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        # Using a higher process noise for line to allow it to adapt faster if the line itself moves/shifts
        # Measurement noise can be lower if the Hough transform is generally precise.
        self.line_kf_process_noise = 1e-1 # Adjust based on how much the line is expected to move frame-to-frame
        self.line_kf_measurement_noise = 1e-2 # Adjust based on accuracy of Hough transform output
        self.line_kf_error_cov_post = 1000.0 # Initial uncertainty

        self.last_known_vertical_line = None # Stores the (x1, y1, x2, y2) of the line
        self.frames_without_line_detection = 0 # Counter for consecutive frames where line is not detected
        self.max_frames_to_persist_line = 5 # How many frames to keep drawing the old line if not detected

        self.frame_count = 0

        # --- ADD THESE NEW ATTRIBUTES WITH DEFAULT VALUES ---
        self.max_frames_to_keep_track = 10 # Number of frames to keep an object track without detection
        self.max_frames_to_persist_line = 30 # Number of frames to predict line without detection
        # --- END NEW ATTRIBUTES ---

    @pyqtSlot()
    def run(self):
        try:
            self.frame_count += 1
            processed_frame, confidence, bbox_count = self.process_frame_with_model()
            self.signals.finished.emit(processed_frame, confidence, bbox_count,
                                       self.tracked_objects, self.next_object_id,
                                       self.vertical_service_line_kf,
                                       self.last_known_vertical_line,
                                       self.frames_without_line_detection)
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
        scaled_boxes = self._scale_boxes(boxes, resized_frame.shape[:2], (original_frame_height, original_frame_width))

        display_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

         # --- Kalman Filter for Object Detections ---
        # Predict all existing tracks
        for track in self.tracked_objects:
            #if track.active:
            #    track.predict()
            track.predict()
            track.active = False

        # Associate new detections with existing tracks
        new_detections = []
        for i in range(len(scaled_boxes)):
            new_detections.append({
                'bbox': scaled_boxes[i],
                'score': scores[i],
                'class_id': class_ids[i]
            })

        self._associate_detections_to_tracks(new_detections)

        # Remove inactive tracks and draw active ones
        # self.tracked_objects = [track for track in self.tracked_objects if track.active and track.frames_since_last_update <= self.max_frames_to_keep_track]
        
        self.tracked_objects = [track for track in self.tracked_objects
                                if track.frames_since_last_update <= self.max_frames_to_keep_track]
        

        # Prepare boxes, scores, and class_ids from tracked objects for drawing
        tracked_boxes = []
        tracked_scores = []
        tracked_class_ids = []
        
        for track in self.tracked_objects:
            if track.active:
                tracked_boxes.append(track.bbox)
                tracked_scores.append(track.score)
                tracked_class_ids.append(track.class_id)

        # Convert lists back to numpy arrays for _draw_detections
        tracked_boxes = np.array(tracked_boxes)
        tracked_scores = np.array(tracked_scores)
        tracked_class_ids = np.array(tracked_class_ids)

        # --- Kalman Filter for Line Detection ---
        detected_lines_dict = self._detect_court_lines(display_frame)
        
        # Process vertical service line with Kalman Filter
        current_vertical_line = detected_lines_dict.get('vertical_service_line')

        
        if current_vertical_line is not None:
            if self.vertical_service_line_kf is None:
                # Initialize Kalman filter for the line
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
                print(f"Frame {self.frame_count}: KF initialized. statePost after init: {self.vertical_service_line_kf.kf.statePost.flatten()}")
            
            # Update Kalman filter with the new measurement
            measurement_line = np.array(current_vertical_line, np.float32).reshape(-1, 1)
            print(f"Frame {self.frame_count}: Measurement before update: {measurement_line.flatten()}")
            print(f"Frame {self.frame_count}: Line KF statePost BEFORE correct: {self.vertical_service_line_kf.kf.statePost.flatten()}")
            corrected_line_state = self.vertical_service_line_kf.update(measurement_line)
            
            #print(f"Frame {self.frame_count}: Detected Line: {current_vertical_line}")
            #print(f"Frame {self.frame_count}: KF Corrected State: {corrected_line_state[:4].flatten()}")

            print(f"Frame {self.frame_count}: KF Corrected State (raw): {corrected_line_state.flatten()}") # Check raw output
            self.last_known_vertical_line = corrected_line_state[:4].flatten().astype(int)
            print(f"Frame {self.frame_count}: KF Corrected State (flattened & int): {self.last_known_vertical_line}")
            self.frames_without_line_detection = 0
        
        else:
            # If no line is detected, predict its position
            if self.vertical_service_line_kf is not None:
                predicted_line_state = self.vertical_service_line_kf.predict()
                print(f"Frame {self.frame_count}: No Detection. KF Predicted State: {predicted_line_state[:4].flatten()}")
                self.last_known_vertical_line = predicted_line_state[:4].flatten().astype(int)
                print(f"Frame {self.frame_count}: No Detection. KF Predicted State (flattened & int): {self.last_known_vertical_line}")
                self.frames_without_line_detection += 1
                
                # If too many frames without detection, stop persisting
                if self.frames_without_line_detection > self.max_frames_to_persist_line:
                    self.last_known_vertical_line = None
                    # Optionally reset KF or keep predicting
                    # self.vertical_service_line_kf = None # Reset KF if you want it to re-initialize on next detection    
        
        # Draw the (Kalman filtered) vertical service line
        if self.last_known_vertical_line is not None:
            x1_line, y1_line, x2_line, y2_line = self.last_known_vertical_line
            cv2.line(display_frame, (x1_line, y1_line), (x2_line, y2_line), (0, 255, 255), 3) # Yellow, thicker
        
        
        
        
        """
        detected_lines_dict = self._detect_court_lines(display_frame)

        for line_name, line_coords in detected_lines_dict.items():
            if line_coords is not None:
                x1, y1, x2, y2 = line_coords
                if line_name == 'vertical_service_line':
                    cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 3) # Yellow, thicker
        
        #processed_frame = self._draw_detections(display_frame, boxes, scores, class_ids)

        #avg_confidence = np.mean(scores) * 100 if len(scores) > 0 else 0
        #bbox_count = len(boxes)
        """

        processed_frame = self._draw_detections(display_frame, tracked_boxes, tracked_scores, tracked_class_ids)

        avg_confidence = np.mean(tracked_scores) * 100 if len(tracked_scores) > 0 else 0
        bbox_count = len(tracked_boxes)

        # cv2.destroyAllWindows() # Close all debug windows before returning

        # Retrieve the line coordinates for analysis
        line_coordinates_for_analysis = None
        if self.last_known_vertical_line is not None:
            x1_line, y1_line, x2_line, y2_line = self.last_known_vertical_line
            line_coordinates_for_analysis = (x1_line, y1_line, x2_line, y2_line)

        self.signals.finished.emit(processed_frame, avg_confidence, bbox_count,
                                   self.tracked_objects, self.next_object_id,
                                   self.vertical_service_line_kf,
                                   line_coordinates_for_analysis, # Pass the coordinates
                                   self.frames_without_line_detection)
        
        return processed_frame, avg_confidence, bbox_count
    
    """
    def _associate_detections_to_tracks(self, detections):
        if not self.tracked_objects: # If no existing tracks, create new ones for all detections
            for det in detections:
                self.tracked_objects.append(TrackedObject(det['bbox'], det['class_id'], det['score'], self.next_object_id))
                self.next_object_id += 1
            return

        # Simple IoU-based association (could be improved with Hungarian algorithm for more robust tracking)
        # For simplicity, we'll use a greedy approach based on distance to predicted location.

        # Mark all existing tracks as unassigned
        for track in self.tracked_objects:
            track.active = False # Will be set to True if assigned or predicted for.

        unassigned_detections_indices = list(range(len(detections)))

        # Store matches (track_idx, detection_idx)
        matches = []

        # Simple greedy matching based on minimum distance
        # You could use a more sophisticated assignment algorithm like Hungarian for better results
        
        # Iterate over tracks and find best matching detection
        for i, track in enumerate(self.tracked_objects):
            best_match_idx = -1
            min_distance = float('inf')

            # Consider only unassigned detections
            for j in list(unassigned_detections_indices): # Iterate over a copy because we'll remove
                det = detections[j]
                
                # Using L2 distance between center points for simplicity
                center_track = ((track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2)
                center_det = ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2)
                
                dist = np.linalg.norm(np.array(center_track) - np.array(center_det))

                # Simple threshold for association based on distance
                # This threshold needs to be tuned based on object movement speed and frame rate
                if dist < min_distance and dist < 100: # Example threshold: 100 pixels, tune this!
                    min_distance = dist
                    best_match_idx = j
            
            if best_match_idx != -1:
                # Assign detection to track
                track.update(detections[best_match_idx]['bbox'], detections[best_match_idx]['score'])
                track.active = True # Mark as active because it was updated
                matches.append((i, best_match_idx))
                unassigned_detections_indices.remove(best_match_idx) # Remove matched detection from unassigned list
            else:
                # If a track wasn't matched with a detection, it still needs to be considered 'active'
                # if its frames_since_last_update is still within the max_frames_to_keep_track limit.
                # Its state was already predicted at the beginning of process_frame_with_model.
                # So, it remains active if it's not too old.
                if track.frames_since_last_update < self.max_frames_to_keep_track:
                    track.active = True # Keep predicting this track even if no detection this frame

        # Create new tracks for unassigned detections
        for idx in unassigned_detections_indices:
            det = detections[idx]
            self.tracked_objects.append(TrackedObject(det['bbox'], det['class_id'], det['score'], self.next_object_id))
            self.next_object_id += 1
    """

    def _associate_detections_to_tracks(self, new_detections):
        """
        Associates current frame's detections with existing tracked objects.
        Improves association by enforcing class consistency and
        prioritizing closest matches.
        """
        if not new_detections:
            return

        # Create lists for assignments
        assigned_detections = [False] * len(new_detections)
        assigned_tracks = [False] * len(self.tracked_objects)

        # Step 1: Prioritize matching based on class_id and proximity
        # Create a list of potential matches: (distance, track_idx, detection_idx)
        potential_matches = []

        for t_idx, track in enumerate(self.tracked_objects):
            # Only consider active tracks (ones that haven't been removed yet)
            if not track.active: # Track's active status reset at start of frame
                continue

            # Get predicted bbox from Kalman Filter for this track
            track_bbox_predicted = track.bbox # This is already the predicted state

            # Calculate centroid for current track
            track_cx = (track_bbox_predicted[0] + track_bbox_predicted[2]) / 2
            track_cy = (track_bbox_predicted[1] + track_bbox_predicted[3]) / 2
            
            for d_idx, detection in enumerate(new_detections):
                if assigned_detections[d_idx]: # Skip already assigned detections
                    continue

                det_bbox = detection['bbox']
                det_class_id = detection['class_id']

                # --- CRITICAL CHANGE 1: Enforce Class Consistency ---
                # A track initiated as a shuttlecock should only match with shuttlecock detections.
                # A track initiated as a racket head should only match with racket head detections.
                if track.class_id != det_class_id:
                    continue # Skip if class IDs don't match

                # Calculate centroid for current detection
                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                det_cy = (det_bbox[1] + det_bbox[3]) / 2

                # Calculate Euclidean distance between centroids
                dist = np.sqrt((track_cx - det_cx)**2 + (track_cy - det_cy)**2)

                # --- Association Threshold (tune this based on observed object speeds and noise) ---
                # This '100' is a placeholder. You'll need to tune it.
                # A good starting point is often 50-150 pixels, depending on resolution.
                if dist < 100: # Tune this threshold!
                    potential_matches.append((dist, t_idx, d_idx))

        # Sort matches by distance (closest first)
        potential_matches.sort(key=lambda x: x[0])

        # Step 2: Assign detections to tracks (Greedy assignment: closest first)
        for dist, t_idx, d_idx in potential_matches:
            if not assigned_tracks[t_idx] and not assigned_detections[d_idx]:
                track = self.tracked_objects[t_idx]
                detection = new_detections[d_idx]

                # Update the track with the new detection
                track.update(detection['bbox'], detection['score'])
                
                assigned_tracks[t_idx] = True
                assigned_detections[d_idx] = True
                
                # Mark track as active for the current frame
                track.active = True # Already done in TrackedObject.update, but good to ensure

        # Step 3: Handle unassigned detections (create new tracks)
        for d_idx, detection in enumerate(new_detections):
            if not assigned_detections[d_idx]:
                # Only create a new track if there isn't already an active track of this class
                # AND we don't already have a track for this class.
                # This is crucial for "only one of each class" requirement.
                # Find if there's an existing active track of the same class
                existing_active_track_of_this_class = False
                for track in self.tracked_objects:
                    if track.active and track.class_id == detection['class_id']:
                        existing_active_track_of_this_class = True
                        break
                
                if not existing_active_track_of_this_class:
                    # Create a new track for this detection
                    new_obj = TrackedObject(
                        bbox=detection['bbox'],
                        class_id=detection['class_id'],
                        score=detection['score'],
                        obj_id=self.next_object_id
                    )
                    self.tracked_objects.append(new_obj)
                    self.next_object_id += 1
                else:
                    # If an active track of this class already exists,
                    # we ignore this new detection for this class.
                    # This helps enforce "only one of each".
                    print(f"DEBUG: Ignoring new detection for class {detection['class_id']} as an active track already exists.")

    
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
            minLineLength=int(height * 0.25), # Slightly increased to favor longer lines. Adjust as needed.
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