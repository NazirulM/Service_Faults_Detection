import numpy as np
import cv2
import torch
from PyQt5.QtCore import (QRunnable, pyqtSlot, pyqtSignal, QObject)
from PyQt5.QtWidgets import QApplication
import sys
from utils.utility import KalmanFilter

class FrameProcessorSignals(QObject):
    # CRITICAL CHANGE: The 4th argument type changed from 'list' to 'object'
    # This is essential to correctly pass the dictionary.
    finished = pyqtSignal(np.ndarray, float, int, object, int, object, tuple, int)
    error = pyqtSignal(str)

class TrackedObject:
    """Represents a single tracked object with its own Kalman filter."""
    def __init__(self, bbox, class_id, score, obj_id):
        self.id = obj_id
        self.class_id = class_id
        # State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.kf = KalmanFilter(state_dim=8, measurement_dim=4,
                               process_noise_cov=1e-1, measurement_noise_cov=0.001, error_cov_post=1000.0)
        
        # Initialize Kalman filter state with current bbox
        initial_state = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0], np.float32).reshape(-1, 1)
        self.kf.kf.statePost = initial_state

        # DEBUG: Print initial state of each new TrackedObject KF
        print(f"TrackedObject {self.id}: KF Initialized. statePost: {self.kf.kf.statePost.flatten()}")
        print(f"TrackedObject {self.id}: KF Initialized. errorCovPost diag: {np.diag(self.kf.kf.errorCovPost)}")
        
        self.bbox = bbox
        self.class_id = class_id
        self.score = score
        self.frames_since_last_update = 0
        self.active = True
        self.matched_this_frame = False # NEW: Flag to indicate if object was matched in current frame

    def update(self, new_bbox, new_score):
        measurement = np.array(new_bbox, np.float32).reshape(-1, 1)
        
        # DEBUG: Print BEFORE update
        print(f"TrackedObject {self.id} (UPDATE): Measurement: {measurement.flatten()}")
        print(f"TrackedObject {self.id} (UPDATE): kf.statePost BEFORE correct: {self.kf.kf.statePost.flatten()}")
        print(f"TrackedObject {self.id} (UPDATE): kf.errorCovPost BEFORE correct (diagonal): {np.diag(self.kf.kf.errorCovPost)}")

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
                 current_frames_without_line_detection=0,shared_cache_list=None, cache_mutex=None
                 ):
        super().__init__()
        self.frame = frame
        self.model = model
        self.threshold = threshold
        self.device = device
        self.img_size = img_size
        self.signals = FrameProcessorSignals()
        
        # Initialize self.tracked_objects as a dictionary.
        # This will now correctly receive a dict if passed, or create an empty one.
        self.tracked_objects = current_tracked_objects if current_tracked_objects is not None else {}
        self.next_object_id = current_next_object_id

        print(f"DEBUG FrameProcessor.__init__: current_vertical_service_line_kf received: {current_vertical_service_line_kf}")

        self.vertical_service_line_kf = current_vertical_service_line_kf
        self.last_known_vertical_line = current_last_known_vertical_line
        self.frames_without_line_detection = current_frames_without_line_detection

        print(f"DEBUG FrameProcessor.__init__: self.vertical_service_line_kf after assignment: {self.vertical_service_line_kf}")
        
        self.shared_cache_list = shared_cache_list
        self.cache_mutex = cache_mutex

        height, width = self.frame.shape[:2]

        self.vertical_service_line_roi = np.array([
            [int(width * 0.491), int(height * 0.552)],
            [int(width * 0.505), int(height * 0.99)]
        ], dtype=np.int32)

    @pyqtSlot()
    def run(self):
        try:
            results = self.model(self.frame, verbose=False, conf=self.threshold/100.0)

            detections = []
            if results and results[0] and results[0].boxes is not None:
                for *xyxy, conf, cls in results[0].boxes.data:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': int(cls)
                    })

            # Filter detections to only accept the best detection per class
            filtered_detections = self.filter_detections_per_class(detections)
            
            # Object Tracking (using Kalman Filters for objects) - this now gets filtered detections
            self._update_and_predict_tracks(filtered_detections)

            # Get the annotated frame directly from results[0].plot()
            annotated_frame = results[0].plot()

            # Service Line Detection and Kalman Filter Update
            best_vertical_line = self._detect_vertical_service_line_and_update_kf(self.frame)
            
            # Draw service line
            if best_vertical_line is not None:
                x1, y1, x2, y2 = map(int, best_vertical_line)
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                if self.last_known_vertical_line is not None:
                    x1, y1, x2, y2 = map(int, self.last_known_vertical_line)
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Prepare data for caching
            tracked_objects_data = []
            # Iterate through values of the dictionary
            for obj in self.tracked_objects.values():
                if obj.kf and obj.kf.kf is not None:
                    tracked_objects_data.append({
                        'id': obj.id,
                        'class_id': obj.class_id,
                        'kf_state': obj.kf.kf.statePost.flatten().tolist() # Caching kf_state here
                    })

            # Use filtered_detections for confidence and bbox_count
            confidence = float(np.mean([d['confidence'] for d in filtered_detections])) if filtered_detections else 0.0
            bbox_count = len(filtered_detections) # This now reflects the count of unique classes with detections

            frame_data_to_cache = {
                'frame': annotated_frame,
                'confidence': confidence, # Use calculated confidence from filtered detections
                'bbox_count': bbox_count, # Use count from filtered detections
                'tracked_objects_data': tracked_objects_data,
                'service_line_coordinates': self.last_known_vertical_line
            }

            if self.shared_cache_list is not None and self.cache_mutex is not None:
                self.cache_mutex.lock()
                try:
                    self.shared_cache_list.append(frame_data_to_cache)
                finally:
                    self.cache_mutex.unlock()

            # Pass the dictionary itself, as the signal type is now 'object'.
            self.signals.finished.emit(
                annotated_frame,
                frame_data_to_cache['confidence'],
                frame_data_to_cache['bbox_count'],
                self.tracked_objects, # Pass the dictionary itself
                self.next_object_id,
                self.vertical_service_line_kf,
                self.last_known_vertical_line,
                self.frames_without_line_detection
            )

        except Exception as e:
            self.signals.error.emit(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def filter_detections_per_class(self, raw_detections):
        """Filters a list of detections to keep only the highest confidence detection for each unique class ID."""
        best_detections_per_class = {}

        for detection in raw_detections:
            class_id = detection['class_id']
            confidence = detection['confidence']

            if class_id not in best_detections_per_class:
                best_detections_per_class[class_id] = detection
            else:
                current_best_confidence = best_detections_per_class[class_id]['confidence']
                if confidence > current_best_confidence:
                    best_detections_per_class[class_id] = detection
        
        return list(best_detections_per_class.values())
    
    def _scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        # Rescales boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
        self._clip_boxes(boxes, img0_shape)
        return boxes

    def _clip_boxes(self, boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2

    def _iou_batch(self, boxes1, boxes2):
        """
        Calculates Intersection over Union (IoU) of boxes.
        Boxes are expected in [x1, y1, x2, y2] format.
        Returns a (N, M) matrix of IoU values where N is len(boxes1) and M is len(boxes2).
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        # Calculate intersection areas
        x_min = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
        y_min = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
        x_max = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
        y_max = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

        intersection_w = np.maximum(0, x_max - x_min)
        intersection_h = np.maximum(0, y_max - y_min)
        intersection_area = intersection_w * intersection_h

        # Calculate union areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1[:, None] + area2 - intersection_area

        # Compute IoU
        iou = intersection_area / (union_area + 1e-6) # Add epsilon to prevent division by zero
        return iou    
    
    def _update_and_predict_tracks(self, current_detections, iou_threshold=0.3, max_frames_to_keep_track=10):
        """
        Updates existing tracks with new detections and creates new tracks for unmatched detections.
        Performs prediction for all active tracks.
        Enforces: At most one active TrackedObject per class_id.
        """
        matched_detection_indices = set()
        
        # Step 1: Predict new locations for all currently tracked objects
        for obj_id in list(self.tracked_objects.keys()): # Iterate over keys to allow deletion during loop
            obj = self.tracked_objects[obj_id]
            obj.predict() # Update obj.bbox with predicted state
            
            # Temporarily mark all existing objects as 'not matched in current frame'
            # This will be updated for matched objects.
            obj.matched_this_frame = False 

        # Dictionary to store the best (matched or new) track for each class_id in this frame
        # This will enforce the "one object per class" rule.
        best_track_for_class_in_frame = {} 

        # Step 2: Associate detections with existing tracks using IoU
        # Prioritize matching filtered_detections to existing tracks
        if current_detections and self.tracked_objects:
            detection_bboxes = [d['bbox'] for d in current_detections]
            track_predicted_bboxes = [obj.bbox for obj in self.tracked_objects.values()]
            track_ids_list = list(self.tracked_objects.keys()) # Keep track of original IDs' order

            if not track_predicted_bboxes:
                iou_matrix = np.zeros((0, len(detection_bboxes)))
            else:
                iou_matrix = self._iou_batch(track_predicted_bboxes, detection_bboxes)

            # Find best matches by sorting IoU in descending order
            sorted_matches_indices = np.argsort(iou_matrix.flatten())[::-1]

            matched_detection_indices = set() # To track which incoming detections have been used

            for flat_idx in sorted_matches_indices:
                track_list_idx, det_idx = np.unravel_index(flat_idx, iou_matrix.shape)
                
                if iou_matrix[track_list_idx, det_idx] >= iou_threshold:
                    current_track_id = track_ids_list[track_list_idx]
                    current_track_obj = self.tracked_objects[current_track_id]
                    current_detection = current_detections[det_idx]
                    
                    # A track can only be matched once, a detection can only be used once,
                    # AND crucially: ensure only one track per class is chosen for this frame
                    if not current_track_obj.matched_this_frame and \
                       det_idx not in matched_detection_indices and \
                       current_track_obj.class_id not in best_track_for_class_in_frame:

                        current_track_obj.update(
                            current_detection['bbox'],
                            current_detection['confidence']
                        )
                        current_track_obj.frames_since_last_update = 0
                        current_track_obj.matched_this_frame = True # Mark as matched for this frame

                        best_track_for_class_in_frame[current_track_obj.class_id] = current_track_obj
                        matched_detection_indices.add(det_idx)
        
        # Step 3: Create new tracks for unmatched *filtered* detections
        for i, detection in enumerate(current_detections):
            if i not in matched_detection_indices:
                # Only create a new track if this class_id doesn't already have a chosen track for this frame
                if detection['class_id'] not in best_track_for_class_in_frame:
                    new_object = TrackedObject(
                        detection['bbox'],
                        detection['class_id'],
                        detection['confidence'],
                        self.next_object_id
                    )
                    best_track_for_class_in_frame[new_object.class_id] = new_object
                    self.next_object_id += 1

        # Step 4: Finalize self.tracked_objects for this frame.
        # This is where we enforce the "one active track per class" rule.
        new_tracked_objects_dict = {}

        # Add all the best tracks chosen in this frame (either matched or newly created)
        for class_id, track_obj in best_track_for_class_in_frame.items():
            new_tracked_objects_dict[track_obj.id] = track_obj
        
        # Now, consider existing tracks that were NOT matched by a detection in this frame.
        # Only keep them if their class_id is not already represented by a matched/new track,
        # AND they haven't exceeded max_frames_to_keep_track.
        for obj_id, obj in self.tracked_objects.items():
            if obj.active and \
               not obj.matched_this_frame and \
               obj.frames_since_last_update <= max_frames_to_keep_track:
                
                # Only add if this class doesn't already have a preferred track from this frame
                if obj.class_id not in new_tracked_objects_dict:
                     new_tracked_objects_dict[obj.id] = obj
                # Else: this 'obj' is a duplicate of a class that DID get a detection,
                # so it will be discarded in favor of the detected one.
            else:
                obj.active = False # Mark as inactive if it's over limit or superseded by a new detection

        self.tracked_objects = new_tracked_objects_dict
        # The self.tracked_objects now only contains active tracks, with at most one per class_id.

    def _detect_vertical_service_line_and_update_kf(self, frame, min_line_length=50, canny_threshold1=50, canny_threshold2=150, hough_threshold=50):
        """
        Detects the most prominent vertical line within a specified ROI using HoughLinesP,
        updates its Kalman filter, and returns the smoothed line coordinates.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

        mask = np.zeros_like(edges)
        x_min, y_min = self.vertical_service_line_roi[0]
        x_max, y_max = self.vertical_service_line_roi[1]
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges, 1, np.pi / 180, hough_threshold,
            minLineLength=min_line_length, maxLineGap=10
        )

        best_vertical_line = None
        max_line_length = 0

        if lines is not None:
            vertical_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < abs(y1 - y2) * 0.5:
                    vertical_lines.append(line[0])
            
            if vertical_lines:
                for x1, y1, x2, y2 in vertical_lines:
                    y1_clipped = max(y1, y_min)
                    y2_clipped = min(y2, y_max)
                    current_length = np.sqrt((x2 - x1)**2 + (y2_clipped - y1_clipped)**2) 
                    
                    if current_length > max_line_length:
                        max_line_length = current_length
                        # Use the original (x1,y1,x2,y2) from HoughLinesP for KF measurement
                        # as KF expects raw, unclipped measurement
                        best_vertical_line = (x1, y1, x2, y2)
        
        if best_vertical_line is not None:
            # Measurement for Kalman Filter: the four coordinates (x1, y1, x2, y2)
            measurement = np.array(best_vertical_line, np.float32).reshape(-1, 1)

            print(f"DEBUG KF: Measured Line from best line: {measurement.flatten()}")

            is_kf_invalid = self.vertical_service_line_kf is None or \
                            (np.sum(np.diag(self.vertical_service_line_kf.kf.errorCovPost)) == 0.0)
            
            if is_kf_invalid:
                from utils.utility import KalmanFilter
                self.vertical_service_line_kf = KalmanFilter(state_dim=4, measurement_dim=4,
                                                            process_noise_cov=1e-5,
                                                            measurement_noise_cov=200.0,
                                                            error_cov_post=1000.0)

                initial_state = measurement
                self.vertical_service_line_kf.kf.statePost = initial_state
                print(f"DEBUG KF Init: Service Line KF Initialized with: {initial_state.flatten()}")

                print(f"DEBUG KF Init: Initialized kf.transitionMatrix:{self.vertical_service_line_kf.kf.transitionMatrix}")
                print(f"DEBUG KF Init: Initialized kf.measurementMatrix:{self.vertical_service_line_kf.kf.measurementMatrix}")
                print(f"DEBUG KF Init: Initialized kf.processNoiseCov:{self.vertical_service_line_kf.kf.processNoiseCov}")
                print(f"DEBUG KF Init: Initialized kf.measurementNoiseCov:{self.vertical_service_line_kf.kf.measurementNoiseCov}")
                print(f"DEBUG KF Init: Initialized kf.errorCovPost:{self.vertical_service_line_kf.kf.errorCovPost}")
            else:
                print(f"DEBUG KF Pre-Update Check: kf.statePost BEFORE predict:{self.vertical_service_line_kf.kf.statePost.flatten()}")
                print(f"DEBUG KF Pre-Update Check: kf.errorCovPost BEFORE predict (diagonal): {np.diag(self.vertical_service_line_kf.kf.errorCovPost)}")
                print(f"DEBUG KF Pre-Update Check: kf.processNoiseCov BEFORE predict:{self.vertical_service_line_kf.kf.processNoiseCov}")

                predicted_state = self.vertical_service_line_kf.predict()
                print(f"DEBUG KF: Predicted/Smoothed State from KF (after predict): {predicted_state.flatten()}")

                print(f"DEBUG KF Pre-Correct Check: kf.statePre BEFORE correct:{self.vertical_service_line_kf.kf.statePre.flatten()}")
                print(f"DEBUG KF Pre-Correct Check: kf.errorCovPre BEFORE correct (diagonal): {np.diag(self.vertical_service_line_kf.kf.errorCovPre)}")
                print(f"DEBUG KF Pre-Correct Check: kf.measurementMatrix BEFORE correct:{self.vertical_service_line_kf.kf.measurementMatrix}")
                print(f"DEBUG KF Pre-Correct Check: kf.measurementNoiseCov BEFORE correct:{self.vertical_service_line_kf.kf.measurementNoiseCov}")

                corrected_state = self.vertical_service_line_kf.update(measurement)
                print(f"DEBUG KF: Corrected State from KF: {corrected_state.flatten()}")
            
            # Use the smoothed (x1, y1, x2, y2) from the Kalman Filter's statePost
            smoothed_x1 = self.vertical_service_line_kf.kf.statePost[0, 0]
            smoothed_y1 = self.vertical_service_line_kf.kf.statePost[1, 0]
            smoothed_x2 = self.vertical_service_line_kf.kf.statePost[2, 0]
            smoothed_y2 = self.vertical_service_line_kf.kf.statePost[3, 0]

            final_y1_roi = y_min
            final_y2_roi = y_max
            
            if abs(smoothed_y2 - smoothed_y1) > 0.01:
                m_line = (smoothed_x2 - smoothed_x1) / (smoothed_y2 - smoothed_y1)
                b_line = smoothed_x1 - m_line * smoothed_y1

                final_x1_roi = m_line * final_y1_roi + b_line
                final_x2_roi = m_line * final_y2_roi + b_line
            else:
                avg_x = (smoothed_x1 + smoothed_x2) / 2.0
                final_x1_roi = avg_x
                final_x2_roi = avg_x

            self.last_known_vertical_line = (int(final_x1_roi), int(final_y1_roi), int(final_x2_roi), int(final_y2_roi))
            self.frames_without_line_detection = 0

            print(f"DEBUG KF: Final line coordinates (last_known_vertical_line): {self.last_known_vertical_line}")

        else: # Line not detected in current frame
            self.frames_without_line_detection += 1
            if self.vertical_service_line_kf is not None:
                if self.frames_without_line_detection <= 5: # Predict for 5 frames max
                    predicted_state = self.vertical_service_line_kf.predict()
                    predicted_x1, predicted_y1, predicted_x2, predicted_y2 = predicted_state.flatten()

                    final_y1_roi = y_min
                    final_y2_roi = y_max
                    
                    if abs(predicted_y2 - predicted_y1) > 0.01:
                        m_line = (predicted_x2 - predicted_x1) / (predicted_y2 - predicted_y1)
                        b_line = predicted_x1 - m_line * predicted_y1
                        final_x1_roi = m_line * final_y1_roi + b_line
                        final_x2_roi = m_line * final_y2_roi + b_line
                    else:
                        avg_x = (predicted_x1 + predicted_x2) / 2.0
                        final_x1_roi = avg_x
                        final_x2_roi = avg_x

                    self.last_known_vertical_line = (int(final_x1_roi), int(final_y1_roi), int(final_x2_roi), int(final_y2_roi))
                else:
                    self.last_known_vertical_line = None
            else:
                self.last_known_vertical_line = None

        return self.last_known_vertical_line