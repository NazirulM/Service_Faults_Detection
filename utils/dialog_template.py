import cv2
import numpy as np

# --- Replicating your KalmanFilter class from utils/utility.py ---
class KalmanFilter:
    def __init__(self, state_dim, measurement_dim, process_noise_cov=1e-2, measurement_noise_cov=1e-1, error_cov_post=1.0):
        self.kf = cv2.KalmanFilter(state_dim, measurement_dim)
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim 
        
        dt = 1.0 # Time step (assuming 1 frame per step)

        # Transition Matrix (A)
        if state_dim == 8:
            self.kf.transitionMatrix = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                                                 [0, 1, 0, 0, 0, dt, 0, 0],
                                                 [0, 0, 1, 0, 0, 0, dt, 0],
                                                 [0, 0, 0, 1, 0, 0, 0, dt],
                                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        elif state_dim == 4:
            self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                                 [0, 1, 0, dt],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        else:
            raise ValueError("Unsupported state_dim")

        # Measurement Matrix (H)
        if measurement_dim == 4 and state_dim == 8:
            self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)
        elif measurement_dim == 2 and state_dim == 4:
            self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        else:
            raise ValueError("Unsupported measurement_dim or state_dim combination")

        # Process Noise Covariance (Q)
        self.kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise_cov

        # Measurement Noise Covariance (R)
        self.kf.measurementNoiseCov = np.eye(measurement_dim, dtype=np.float32) * measurement_noise_cov
        
        # Error Covariance (P) - initialized after correction
        self.kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * error_cov_post 

        # Give higher initial uncertainty to position components
        if state_dim == 8:
            for i in range(4): # x1, y1, x2, y2
                self.kf.errorCovPost[i,i] = 10000.0 # High uncertainty
            for i in range(4, 8): # vx1, vy1, vx2, vy2
                self.kf.errorCovPost[i,i] = 100.0 # Lower uncertainty
        elif state_dim == 4:
            self.kf.errorCovPost[0,0] = 10000.0 # x
            self.kf.errorCovPost[1,1] = 10000.0 # y
            self.kf.errorCovPost[2,2] = 100.0 # vx
            self.kf.errorCovPost[3,3] = 100.0 # vy


    def predict(self):
        return self.kf.predict()

    def update(self, measurement):
        if measurement.shape != (self.measurement_dim, 1):
            raise ValueError(f"Measurement has incorrect shape: {measurement.shape}. Expected ({self.measurement_dim}, 1).")
        if measurement.dtype != np.float32:
            measurement = measurement.astype(np.float32)

        print(f"DEBUG KF Update: Q diagonal: {np.diag(self.kf.processNoiseCov)}")
        print(f"DEBUG KF Update: R diagonal: {np.diag(self.kf.measurementNoiseCov)}")
        print(f"DEBUG KF Update: Transition Matrix (A):\n{self.kf.transitionMatrix}")
        print(f"DEBUG KF Update: Measurement Matrix (H):\n{self.kf.measurementMatrix}")
        print(f"DEBUG KF Update: kf.statePost BEFORE correct: {self.kf.statePost.flatten()}")
        print(f"DEBUG KF Update: kf.errorCovPost BEFORE correct: {np.diag(self.kf.errorCovPost)}")
        print(f"DEBUG KF Update: Measurement received: {measurement.flatten()}")

        # Perform prediction first (important for correct to work as expected)
        self.kf.predict() 
        
        corrected_state = self.kf.correct(measurement)
        
        print(f"DEBUG KF Update: kf.statePost AFTER correct: {self.kf.statePost.flatten()}")
        print(f"DEBUG KF Update: kf.errorCovPost AFTER correct: {np.diag(self.kf.errorCovPost)}")

        # Check for NaN/Inf
        if np.any(np.isnan(self.kf.statePost)) or np.any(np.isinf(self.kf.statePost)):
            print("WARNING: statePost contains NaN or Inf after correct!")
        if np.any(np.isnan(self.kf.errorCovPost)) or np.any(np.isinf(self.kf.errorCovPost)):
            print("WARNING: errorCovPost contains NaN or Inf after correct!")

        return corrected_state

# --- End KalmanFilter class ---


if __name__ == "__main__":
    print("\n--- Running Isolated Kalman Filter Test (App Context) ---")

    # These are the actual parameters you pass from FrameProcessor.py
    # to the KalmanFilter initialization for the vertical_service_line_kf
    STATE_DIM = 8
    MEASUREMENT_DIM = 4
    LINE_KF_PROCESS_NOISE = 1e-2 # From FrameProcessor
    LINE_KF_MEASUREMENT_NOISE = 1e-1 # From FrameProcessor
    LINE_KF_ERROR_COV_POST = 1.0 # This default is then overridden for positions/velocities

    # 1. Initialize Kalman Filter object exactly as in FrameProcessor
    vertical_service_line_kf = KalmanFilter(
        state_dim=STATE_DIM, 
        measurement_dim=MEASUREMENT_DIM,
        process_noise_cov=LINE_KF_PROCESS_NOISE,
        measurement_noise_cov=LINE_KF_MEASUREMENT_NOISE,
        error_cov_post=LINE_KF_ERROR_COV_POST # This will be adjusted in init
    )

    # 2. Simulate the FIRST DETECTED LINE from your output
    # Frame 1: Measurement before update: [ 961 596 944 1069]
    first_detected_line_measurement = np.array([961.0, 596.0, 944.0, 1069.0], np.float32).reshape(-1, 1)

    # 3. Simulate the INITIAL statePost you're seeing in your logs
    # Frame 1: KF initialized. statePost after init: [ 961 596 944 1069 0 0 0 0]
    initial_kf_state = np.array([961.0, 596.0, 944.0, 1069.0, 0.0, 0.0, 0.0, 0.0], np.float32).reshape(-1, 1)
    vertical_service_line_kf.kf.statePost = initial_kf_state

    print(f"Manual initial statePost set: \n{vertical_service_line_kf.kf.statePost.flatten()}")
    print(f"Manual initial errorCovPost diagonal (from KF object): \n{np.diag(vertical_service_line_kf.kf.errorCovPost)}")


    print("\n--- Performing first predict/correct cycle ---")
    
    # Crucially, ensure that predict is called before correct!
    # Although kf.correct() internally calls predict() if kf.statePre is not set,
    # it's good practice to call it explicitly or ensure it's handled.
    # Our update method calls predict().
    
    corrected_state_output = vertical_service_line_kf.update(first_detected_line_measurement)
    
    print(f"\nFinal corrected_state returned by update method: \n{corrected_state_output.flatten()}")
    print(f"Final kf.statePost from object: \n{vertical_service_line_kf.kf.statePost.flatten()}")
    print(f"Final kf.errorCovPost diagonal from object: \n{np.diag(vertical_service_line_kf.kf.errorCovPost)}")