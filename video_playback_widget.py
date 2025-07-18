from PyQt5.QtWidgets import (
    QStackedWidget, QWidget, QLabel, QPushButton, QVBoxLayout, 
    QGroupBox, QHBoxLayout, QFileDialog, QSlider,
    QProgressDialog, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThreadPool, QThread, pyqtSlot, QMutex
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QDragEnterEvent, QDropEvent
from utils.utility import ModelWarmupWorker
from video_processer import FrameProcessor
import time
import cv2
from collections import deque
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import torch
import numpy as np
from ultralytics import YOLO
import torch


class VideoPlaybackWidget(QWidget):
    def __init__(self, local_processing):
        super().__init__()
        self.local_processing = local_processing
        self.video_path = ""
        self.cap = None
        # Two distinct timers: one for processing frames, one for playing back cached frames
        self.processing_timer = QTimer()
        self.playback_timer = QTimer()

        self.current_frame = None
        self.fps = 30
        self.frame_history = deque(maxlen=50)  # Store last 50 frames for chart

        # Enhanced GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load YOLOv8 model
        self.model = YOLO("runs/detect/yolov8n_custom/weights/best.pt")  # Make sure this path is correct
        self.model.to(self.device)

        # Access the class names
        class_names_mapping = self.model.names
        print(f"Class names mapping: {class_names_mapping}")

        if torch.cuda.is_available():
            #self.model = self.model.cuda()
            torch.backends.cudnn.benchmark = True
        
        # Thread pool setup
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(2)  # Adjust based on your hardware
        self.pending_frames = 0
        #self.last_processed_frame = None

        print("cv2 version: " + cv2.__version__)

         # --- Caching Mechanism ---
        self.cached_frames_data = deque() # Stores processed frames: {'frame': np.ndarray, 'confidence': float, 'bbox_count': int}
        self.cache_mutex = QMutex() # Mutex to protect access to self.cached_frames_data
        self.current_playback_index = 0 # Index for playing back from cache
        self.total_video_frames = 0 # Total frames in the video
        self.all_frames_processed_and_cached = False # Flag to indicate completion of processing phase
        # --- End Caching Mechanism ---
        
        self.setupUI()
        
        # Video info tracking
        self.measured_fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.bounding_box_count = 0  # Will be updated by your model

        # Thread management
        self.warmup_thread = None  # Track our warmup thread
        self.warmup_worker = None
        self.loading_dialog = None

        # Model warmup flag
        self.model_ready = False

        # New flag for KF warmup state
        self.kalman_filters_warmed_up = False

        # --- NEW: PERSISTENT TRACKING STATE VARIABLES ---
        self.persistent_tracked_objects = {}
        self.persistent_next_object_id = 0
        self.persistent_vertical_service_line_kf = None # Store the KalmanFilter object itself
        self.persistent_last_known_vertical_line = None # Stores the (x1, y1, x2, y2)
        self.persistent_frames_without_line_detection = 0
        # --- END NEW PERSISTENT TRACKING STATE VARIABLES ---

        # Store the last processed frame data for immediate display during processing
        self.last_processed_frame_data = None

    def setupUI(self):
        # Main horizontal layout for left and right panels
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Left Panel (Video Playback)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Back button
        back_btn = QPushButton("← Back to Upload")
        back_btn.clicked.connect(self.local_processing.go_back_to_upload)
        left_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        
        # Video display
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 360)
        self.video_display.setStyleSheet("background-color: black; color: white;")
        left_layout.addWidget(self.video_display, stretch=1)
        
        # Right Panel (Analysis Dashboard)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Confidence Threshold Slider
        threshold_group = QGroupBox("Confidence Threshold")
        threshold_group.setObjectName("box_title")
        threshold_layout = QVBoxLayout(threshold_group)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(35)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel(f"Current Threshold: 35%")
        self.threshold_label.setObjectName("description_label")
        threshold_layout.addWidget(self.threshold_label)
        right_layout.addWidget(threshold_group)
        
        # Matplotlib Figure for Line Chart
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Confidence Over Time")
        self.ax.set_xlabel("Frames")
        self.ax.set_ylabel("Confidence %")
        self.ax.set_ylim(0, 100)
        self.line, = self.ax.plot([], [])
        self.canvas.setMinimumHeight(200)
        right_layout.addWidget(self.canvas)
        
        # Video Information Panel (replaces Violation Summary)
        info_group = QGroupBox("Video Information")
        info_group.setObjectName("box_title")
        info_layout = QVBoxLayout(info_group)
        
        # FPS display
        self.fps_label = QLabel("Playback FPS: 0.0")
        self.fps_label.setObjectName("description_label")
        
        # Bounding box count display
        self.bbox_label = QLabel("Bounding Boxes: 0")
        self.bbox_label.setObjectName("description_label")
        
        # Video resolution display
        self.resolution_label = QLabel("Resolution: N/A")
        self.resolution_label.setObjectName("description_label")
        
        # Frame count display
        self.frame_count_label = QLabel("Frame: 0")
        self.frame_count_label.setObjectName("description_label")

        self.status_label = QLabel("Status: Idle") # New status label
        self.status_label.setObjectName("description_label")
        
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.bbox_label)
        info_layout.addWidget(self.resolution_label)
        info_layout.addWidget(self.frame_count_label)
        info_layout.addWidget(self.status_label) # Add status label
        right_layout.addWidget(info_group)
        
        # Status Dashboard (can keep this or modify as needed)
        self.status_dashboard = QLabel()
        self.status_dashboard.setAlignment(Qt.AlignCenter)
        self.status_dashboard.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                border-radius: 10px;
            }
        """)
        self.status_dashboard.setText("LEGAL SERVICE")
        self.status_dashboard.setStyleSheet("background-color: #e6f3ff; color: #0066cc;")
        right_layout.addWidget(self.status_dashboard)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)

    def load_video(self, file_path):
        self.video_path = file_path

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            # Handle error, e.g., show a QMessageBox, disable controls
            self.actual_video_fps = 0 # Default to 0 or a safe value
            self.local_processing.go_back_to_upload() # Go back if video fails to open
            return
        else:
            self.actual_video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Detected video FPS: {self.actual_video_fps}")
        
        # --- Reset all persistent state when loading a new video ---
        self.processing_timer.stop()
        self.playback_timer.stop()

        # --- NEW: Reset persistent state when loading a new video ---
        self.persistent_tracked_objects = {}
        self.persistent_next_object_id = 0
        self.persistent_vertical_service_line_kf = None
        self.persistent_last_known_vertical_line = None
        self.persistent_frames_without_line_detection = 0
        self.cached_frames_data.clear() # Clear cache for new video

        self.current_playback_index = 0
        self.all_frames_processed_and_cached = False

        self.model_ready = False # Reset model ready flag
        self.kalman_filters_warmed_up = False # Reset KF warmup flag
        # --- END NEW ---

        # Initialize dialog with proper parent and settings
        self.loading_dialog = QProgressDialog(
            "Initializing model...", 
            "Cancel", 
            0, 
            100, 
            self  # Critical parent parameter
        )
        self.loading_dialog.setWindowTitle("Processing Status")
        self.loading_dialog.setWindowModality(Qt.WindowModal)
        #self.loading_dialog.setAttribute(Qt.WA_DeleteOnClose, False)  # Prevent auto-deletion
        #self.loading_dialog.canceled.connect(self.handle_cancel)  # Proper signal connection
        self.loading_dialog.setCancelButton(None)  # Disable cancel button initially
        self.loading_dialog.setAutoClose(False)
        self.loading_dialog.setAutoReset(False)

        # Show loading dialog before processing starts
        self.loading_dialog.setLabelText("Loading video and initializing model...")
        self.loading_dialog.setValue(0)
        self.loading_dialog.show()

        """
        # Properly manage warmup thread
        if self.warmup_thread and self.warmup_thread.isRunning():
            self.warmup_thread.quit()
        """

        # Warm up model in background
        #self.warmup_model()

        # Start warmup for model and KFs
        self.warmup_model_and_kfs()
        
        """
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("Error opening video file")
            return
        """
        
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution_label.setText(f"Resolution: {width}x{height}")
        
        # Reset counters
        self.frame_count = 0
        self.start_time = time.time()
        self.bounding_box_count = 0
        self.update_info_display()

        """
        self.timer.stop()    
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // self.fps) 
        """ 
        
        # Reset frame history
        self.frame_history.clear()
        self.update_chart() # Clear chart display

        
    
    def warmup_model_and_kfs(self):
        
        self.cleanup_warmup()
            
        # Run in a separate thread to keep UI responsive
        self.warmup_thread = QThread()
        self.warmup_worker = ModelWarmupWorker(
            self.model,
            initial_line_kf_state=self.persistent_vertical_service_line_kf,
            initial_tracked_objects_state=self.persistent_tracked_objects,
            initial_next_object_id=self.persistent_next_object_id
        )
        self.warmup_worker.moveToThread(self.warmup_thread)
        self.warmup_worker.progress.connect(self.loading_dialog.setValue)
        self.warmup_worker.finished.connect(self.on_warmup_complete)
        self.warmup_worker.error.connect(self.on_warmup_error)
        #self.warmup_worker.progress.connect(self.loading_dialog.setValue)
        self.warmup_thread.started.connect(self.warmup_worker.run)
        self.warmup_thread.start()

        """
        # Run dummy inference to initialize model
        if self.loading_dialog:
            self.loading_dialog.setLabelText("Warming up model...")
            # Enable cancel button only after dialog is shown
            self.loading_dialog.setCancelButtonText("Cancel")
            self.loading_dialog.canceled.connect(self.handle_cancel)    
        """

        self.loading_dialog.setLabelText("Warming up model and Kalman filters...")
        self.loading_dialog.setCancelButtonText("Cancel")
        self.loading_dialog.canceled.connect(self.handle_cancel)
        

    def cleanup_warmup(self):
        """Safely clean up warmup resources"""
        if hasattr(self, 'warmup_worker') and self.warmup_worker:
            self.warmup_worker.stop()
            # Disconnect signals to prevent crashes if thread is already exiting
            try:
                self.warmup_worker.finished.disconnect(self.on_warmup_complete)
                self.warmup_worker.error.disconnect(self.on_warmup_error)
                self.warmup_worker.progress.disconnect(self.loading_dialog.setValue)
            except TypeError: # Signal might already be disconnected
                pass
        
        if hasattr(self, 'warmup_thread') and self.warmup_thread:
            if self.warmup_thread.isRunning():
                self.warmup_thread.quit()
                # Give it a short time to finish, but don't block
                # QTimer.singleShot(300, self.warmup_thread.deleteLater)
                self.warmup_thread.wait(2000)
            else:
                self.warmup_thread.deleteLater()
        
        """
        # Clear references
        if hasattr(self, 'warmup_worker'):
            del self.warmup_worker
        if hasattr(self, 'warmup_thread'):
            del self.warmup_thread
        """

        # Clear references to allow garbage collection
        self.warmup_worker = None
        self.warmup_thread = None

    
    def cancel_loading(self):
        """Proper cancellation handling"""
        if self.warmup_thread and self.warmup_thread.isRunning():
            self.warmup_thread.quit()
        
        if self.loading_dialog:
            self.loading_dialog.close()
            self.loading_dialog = None
            
        # Optional: go back to upload screen
        self.local_processing.go_back_to_upload()

    @pyqtSlot(object, object, int) # Signal from ModelWarmupWorker now includes updated KFs/objects
    def on_warmup_complete(self, warmed_up_line_kf, warmed_up_tracked_objects, warmed_up_next_id):
        self.model_ready = True
        self.kalman_filters_warmed_up = True # Mark KFs as warmed up

        # --- CRITICAL: Update persistent KF states with the warmed-up versions ---
        self.persistent_vertical_service_line_kf = warmed_up_line_kf
        self.persistent_tracked_objects = warmed_up_tracked_objects
        self.persistent_next_object_id = warmed_up_next_id
        # --- END CRITICAL UPDATE ---

        self.loading_dialog.setValue(100)
        self.loading_dialog.setLabelText("Model ready! Starting video processing...")
        # Only close the dialog, not the parent widget
        QTimer.singleShot(1000, self.loading_dialog.hide)

        # Start the processing phase
        self.start_processing()

    def on_warmup_error(self, error_msg):
        self.loading_dialog.cancel()
        QMessageBox.critical(self, "Model Error", f"Failed to initialize model:\n{error_msg}")
        self.local_processing.go_back_to_upload() # Go back on critical error
    
    def cancel_processing(self):
        """Handle user cancellation"""
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.model_ready = False
        self.loading_dialog.cancel()

    def safe_close_dialog(self):
        if self.loading_dialog:
            self.loading_dialog.hide()
            # self.loading_dialog.deleteLater()
            self.loading_dialog = None
    
    def handle_cancel(self):
        #Handles user cancellation during loading/warmup.
        print("Warmup/Processing cancelled by user.")
        self.timer.stop() # Stop any active video timer
        if self.cap and self.cap.isOpened():
            self.cap.release() # Release video resource
        self.cleanup_warmup() # Ensure warmup thread is stopped
        self.safe_close_dialog()
        self.local_processing.go_back_to_upload() # Go back to previous screen

    def start_processing(self):
        """
        Initiates the processing phase: reading frames from video and dispatching
        them to background threads for processing and caching.
        """
        if not self.model_ready or not self.kalman_filters_warmed_up:
            QMessageBox.warning(self, "Processing Not Ready", "Model or Kalman Filters not yet warmed up.")
            return

        self.status_label.setText("Status: Processing frames...")
        # Reset video capture to the beginning for processing
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0 # Reset frame counter for processing
        self.start_time = time.time() # Reset start time for processing FPS calculation
        self.pending_frames = 0
        self.cached_frames_data.clear() # Ensure cache is empty before new processing

        # Start a timer to continuously push frames for processing
        self.processing_timer.timeout.connect(self.process_and_cache_frame)
        self.processing_timer.start(1) # Start as fast as possible to push frames to threads (minimal interval)

    def process_and_cache_frame(self):
        """
        Reads a frame from the video and dispatches it for processing and caching
        in a background thread.
        """
        # If all frames have been read and dispatched, stop the processing timer
        if self.frame_count >= self.total_video_frames:
            self.processing_timer.stop()
            self.status_label.setText(f"Status: Waiting for {self.pending_frames} frames to cache...")
            print("All raw frames dispatched for processing.")
            self.check_processing_completion() # Check if all finished (might be 0 pending already)
            return

        # Limit pending tasks to prevent overwhelming the system.
        # Allow more pending tasks than max threads to keep threads busy.
        if self.pending_frames >= self.threadpool.maxThreadCount() * 2:
            # print(f"DEBUG: Too many pending frames ({self.pending_frames}), pausing frame reading.")
            return

        ret, frame = self.cap.read()
        if not ret:
            # This case should ideally be caught by `self.frame_count >= self.total_video_frames`
            # but acts as a safeguard.
            self.processing_timer.stop()
            print("Error: Could not read frame during processing phase (unexpected end of video).")
            self.check_processing_completion()
            return

        # Create a FrameProcessor QRunnable instance
        # IMPORTANT: Pass a SNAPSHOT of the current persistent state variables to the worker.
        # The worker will operate on these copies and return the NEW state.
        # Deepcopy mutable objects (like TrackedObject list) to ensure isolation in worker.
        # KalmanFilter objects are inherently mutable and will be passed by reference,
        # but the logic for updating them is designed to be sequential (one worker updates and returns new state).
        processor = FrameProcessor(
            frame=frame,
            model=self.model,
            threshold=self.threshold_slider.value(),
            device=self.device,
            img_size=640, # Assumed fixed model input size
            current_tracked_objects=self.persistent_tracked_objects,
            current_next_object_id=self.persistent_next_object_id,
            current_vertical_service_line_kf=self.persistent_vertical_service_line_kf,
            current_last_known_vertical_line=self.persistent_last_known_vertical_line,
            current_frames_without_line_detection=self.persistent_frames_without_line_detection,
            shared_cache_list=self.cached_frames_data, # Reference to the shared cache (deque)
            cache_mutex=self.cache_mutex # Reference to the shared mutex
        )

        processor.signals.finished.connect(self.on_processing_finished)
        processor.signals.error.connect(self.on_processing_error)
        self.threadpool.start(processor)

        self.frame_count += 1
        self.pending_frames += 1
        self.update_info_display()
        
        # Update progress dialog if it's still visible
        if self.loading_dialog and self.loading_dialog.isVisible():
            progress_value = int(self.frame_count / self.total_video_frames * 100)
            self.loading_dialog.setValue(progress_value)
            self.loading_dialog.setLabelText(f"Processing frames... {self.frame_count}/{self.total_video_frames}")
        
        self.status_label.setText(f"Status: Processing ({self.frame_count}/{self.total_video_frames})")
    
    def update_frame(self):

        if not self.model_ready or not self.kalman_filters_warmed_up:
            # If model or KFs are not ready, don't attempt to process.
            # Keep the UI responsive by just returning.
            # You might display a "Loading..." message on the video_display itself.
            return
        
       # Check if we have processed frames in cache to display immediately
        # This makes playback smoother if processing is slower than display FPS
        # However, for continuous processing, we often just display the *last completed* frame.
        # For simplicity and to stick to your direct request, we prioritize real-time processing.
        # If pending frames queue is full, we temporarily display the last processed frame
        # to avoid stuttering while waiting for background tasks to free up.
        if self.pending_frames >= self.threadpool.maxThreadCount():
            # Too many frames pending, just re-display the last successful frame
            if self.last_processed_frame:
                self.display_frame(
                    self.last_processed_frame[0],
                    self.last_processed_frame[1],
                    self.last_processed_frame[2]
                )
            return
        

        # Only process if threadpool has available threads
        if self.threadpool.activeThreadCount() < self.threadpool.maxThreadCount():
            ret, frame = self.cap.read()
            if not ret:
                self.frame_count = 0    
                self.pending_frames = 0
                # Video ended - reset to start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.timer.stop()
                    if hasattr(self, 'loading_dialog') and self.loading_dialog:
                        self.loading_dialog.hide()
                    return
            
            # Create and start processing task
            processor = FrameProcessor(frame, self.model, 
                                    self.threshold_slider.value(),
                                    self.device, img_size=640, current_tracked_objects=self.persistent_tracked_objects, # <-- Pass the current list
                                    current_next_object_id=self.persistent_next_object_id,   # <-- Pass the current ID
                                    current_vertical_service_line_kf=self.persistent_vertical_service_line_kf,
                                    current_last_known_vertical_line=self.persistent_last_known_vertical_line,
                                    current_frames_without_line_detection=self.persistent_frames_without_line_detection,
                                    )
            
            processor.signals.finished.connect(self.on_processing_finished)
            processor.signals.error.connect(self.on_processing_error)
            self.threadpool.start(processor)
            
            self.frame_count += 1
            self.pending_frames += 1
            
            
            # Update info display
            self.update_info_display()
   
    # --- NEW: SLOT TO RECEIVE AND UPDATE PERSISTENT STATE ---
    @pyqtSlot(np.ndarray, float, int, object, int, object, tuple, int)
    def on_processing_finished(self, processed_frame, confidence, bbox_count,
                               updated_tracked_objects, updated_next_object_id,
                               updated_line_kf, updated_last_known_line,
                               updated_frames_without_line):
        self.pending_frames -= 1
        self.bounding_box_count = bbox_count
        self.last_processed_frame = (processed_frame, confidence, bbox_count)
        self.display_frame(processed_frame, confidence, bbox_count)

        # --- CRITICAL: Update the persistent state with the results from the worker thread ---
        self.persistent_tracked_objects = updated_tracked_objects
        self.persistent_next_object_id = updated_next_object_id
        self.persistent_vertical_service_line_kf = updated_line_kf
        self.persistent_last_known_vertical_line = updated_last_known_line
        self.persistent_frames_without_line_detection = updated_frames_without_line
        # --- END CRITICAL UPDATE ---

        self._print_cached_frames_details() # Call to print the full cache

        # Check if all frames have been processed and cached
        self.check_processing_completion()

    def on_processing_error(self, error_msg):
        self.pending_frames -= 1
        print(f"Frame processing error: {error_msg}")
        self.loading_dialog.setLabelText(f"Error: {error_msg}")
        #QTimer.singleShot(2000, lambda: self.loading_dialog.setLabelText("Resuming processing..."))
        self.check_processing_completion() # Still check completion, in case it's the last frame
    
    def check_processing_completion(self):
        """
        Checks if all frames have been read/dispatched AND all dispatched frames
        have completed processing and caching. If so, it transitions to the playback phase.
        """
        # Ensure all frames have been *dispatched* AND all dispatched frames have *finished* processing
        if self.frame_count >= self.total_video_frames and self.pending_frames == 0:
            if not self.all_frames_processed_and_cached: # Only run once when processing is truly complete
                self.all_frames_processed_and_cached = True
                print("All video frames processed and cached. Starting playback.")
                self.status_label.setText("Status: All frames cached. Starting playback...")
                
                # Close the processing dialog if it's still open
                QTimer.singleShot(500, self.safe_close_dialog) 
                # Start playback after a slight delay to ensure dialog closes smoothly
                QTimer.singleShot(1000, self.start_playback)
    
    def start_playback(self):
        """
        Initiates the video playback phase, displaying frames from the cache.
        """
        if not self.all_frames_processed_and_cached:
            QMessageBox.warning(self, "Playback Not Ready", "Frames not yet fully processed and cached.")
            return
        
        if not self.cached_frames_data:
            QMessageBox.critical(self, "Playback Error", "Cached frames data is empty. Cannot start playback.")
            self.local_processing.go_back_to_upload()
            return

        self.status_label.setText("Status: Playing from cache")
        self.current_playback_index = 0 # Start from the beginning of the cache for playback
        
        self.playback_timer.timeout.connect(self.display_cached_frame)
        self.playback_timer.start(1000 // self.fps) # Use target FPS for playback
    
    def display_cached_frame(self):
        """
        Displays the next frame from the cached data. Loops playback.
        """
        self.cache_mutex.lock() # Acquire mutex before accessing cache
        try:
            if not self.cached_frames_data:
                print("Cache is empty, cannot play back.")
                self.playback_timer.stop()
                self.status_label.setText("Status: Cache Empty")
                return

            if self.current_playback_index >= len(self.cached_frames_data):
                self.current_playback_index = 0 # Loop back to the beginning

            frame_data = self.cached_frames_data[self.current_playback_index]
            processed_frame = frame_data['frame']
            confidence = frame_data['confidence']
            bbox_count = frame_data['bbox_count']
        finally:
            self.cache_mutex.unlock() # Release mutex

        self.display_frame(processed_frame, confidence, bbox_count)

        self.current_playback_index += 1
        # Update frame count label for playback
        # Subtract 1 from current_playback_index because it's incremented *after* display
        display_idx = self.current_playback_index if self.current_playback_index > 0 else len(self.cached_frames_data)
        self.frame_count_label.setText(f"Frame: {display_idx}/{len(self.cached_frames_data)} (Cached)")

    def display_frame(self, processed_frame, confidence, bbox_count):
        # Update frame history for chart
        self.frame_history.append(confidence)
        self.update_chart()

        # Convert to QImage and display
        # Ensure processed_frame is RGB for QImage.Format_RGB888
        if processed_frame.shape[2] == 3: # If 3 channels, assume BGR from OpenCV and convert to RGB
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        else: # Grayscale or other formats might need different handling
            rgb_frame = processed_frame # Or convert appropriately if not RGB
        
        #rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage and display
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_display.setPixmap(QPixmap.fromImage(q_img))

        # Scale pixmap to fit the QLabel while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_display.setPixmap(scaled_pixmap)

    def _print_cached_frames_details(self):
        """
        Iterates through the self.cached_frames_data deque and prints details
        for each cached frame.
        """
        print("\n--- Current Cached Frames Data ---")
        # Acquire mutex to safely access the shared deque
        self.cache_mutex.lock()
        try:
            if not self.cached_frames_data:
                print("  Cache is empty.")
            else:
                for i, frame_data in enumerate(self.cached_frames_data):
                    frame_np = frame_data.get('frame')
                    confidence = frame_data.get('confidence')
                    bbox_count = frame_data.get('bbox_count')
                    tracked_objects_data = frame_data.get("tracked_objects_data")
                    service_line_coordinates = frame_data.get("service_line_coordinates")

                    print(f"  Frame {i+1}:")
                    if frame_np is not None:
                        print(f"    Shape: {frame_np.shape}, Dtype: {frame_np.dtype}")
                        # You can print a small snippet of the array if desired, e.g., the value of a specific pixel
                        # print(f"    Top-left pixel value: {frame_np[0,0]}")
                    else:
                        print(f"    (Frame np.ndarray is None)")
                    
                    if confidence is not None:
                        print(f"    Confidence: {confidence:.2f}")
                    if bbox_count is not None:
                        print(f"    Bounding Box Count: {bbox_count}")
                    if tracked_objects_data is not None:
                        print(f"    Tracked Objects: {tracked_objects_data}")
                    if service_line_coordinates is not None:
                        print(f"    Service Line Coordinates: {service_line_coordinates}")
        finally:
            # Release mutex
            self.cache_mutex.unlock()
        print("----------------------------------\n")

    def process_frame(self, frame):
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Example processing - replace with your actual analysis
        # Here we just return random values for demonstration
        confidence = np.random.randint(0, 100)
        bbox_count = np.random.randint(0, 5)  # Random number of boxes (0-4)
        
        # Draw some example annotations
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Boxes: {bbox_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw example bounding boxes (random positions)
        for i in range(bbox_count):
            x1, y1 = np.random.randint(0, frame.shape[1]//2), np.random.randint(0, frame.shape[0]//2)
            x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(50, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame, confidence, bbox_count


    def update_info_display(self):
        # Update FPS based on current phase
        if self.processing_timer.isActive():
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                processing_fps = self.frame_count / elapsed
                self.fps_label.setText(f"Proc. FPS: {processing_fps:.1f}")
            else:
                self.fps_label.setText("Proc. FPS: 0.0")
        elif self.playback_timer.isActive():
             self.fps_label.setText(f"Play. FPS: {self.actual_video_fps:.1f}") # Display video's original FPS for playback
        else:
            self.fps_label.setText("FPS: 0.0")

        self.bbox_label.setText(f"Bounding Boxes: {self.bounding_box_count}")

    def update_chart(self):
        self.line.set_data(range(len(self.frame_history)), list(self.frame_history))
        self.ax.set_xlim(0, len(self.frame_history))
        self.canvas.draw()

    def update_threshold(self, value):
        self.threshold_label.setText(f"Current Threshold: {value}%")

    def closeEvent(self, event):
        if self.warmup_thread and self.warmup_thread.isRunning():
            self.warmup_thread.quit()
            self.warmup_thread.wait(1000)
        
        # Stop all timers
        self.processing_timer.stop()
        self.playback_timer.stop()

        if self.cap and self.cap.isOpened():
            self.cap.release()

        if self.timer.isActive():
            self.timer.stop()

        self.safe_close_dialog()
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Additional cleanup for shared memory

        # Clean up warmup resources
        self.cleanup_warmup()

        # Clean up processing thread pool (optional, but good practice)
        if self.threadpool.activeThreadCount() > 0:
            print("Waiting for processing threads to finish...")
            self.threadpool.waitForDone(2000) # Wait up to 2 seconds for active tasks to complete
        
         # Close loading dialog if open
        self.safe_close_dialog()

        event.accept()

