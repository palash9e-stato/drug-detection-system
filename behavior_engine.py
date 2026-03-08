
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class BehaviorAnalyzer:
    def __init__(self):
        # Initialize PoseLandmarker using Tasks API
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        
        # We need drawing utils... but mp.solutions.drawing_utils might be missing too?
        # If missing, we implement simple drawing.
        self.has_drawing_utils = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'drawing_utils')
        if self.has_drawing_utils:
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
        
        # State tracking
        self.history = []
        self.jitter_scores = []
        self.baseline_pose = None
        self.start_time = time.time()
        
    def analyze_frame(self, frame):
        """
        Analyzes a frame for behavioral cues.
        Returns: annotated_frame, metrics_dict
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        detection_result = self.landmarker.detect(mp_image)
        
        # Render back to BGR for display
        annotated_image = frame.copy()
        
        metrics = {
            "status": "Neutral",
            "stress_level": 0.0,
            "surrender_detected": False,
            "aggression_detected": False,
            "anomalies": []
        }
        
        if detection_result.pose_landmarks:
            # We process the first pose
            landmarks = detection_result.pose_landmarks[0]
            
            # Map landmarks (New API returns NormalizedLandmark objects)
            # Indices are same as legacy: 11=Left Shoulder, 12=Right Shoulder, 15=Left Wrist, 16=Right Wrist
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]
            
            # 1. Logic: Hands Up (Surrender)
            # y is normalized [0,1], 0 is top.
            if l_wrist.y < l_shoulder.y and r_wrist.y < r_shoulder.y:
                metrics["surrender_detected"] = True
                metrics["status"] = "SURRENDER / COMPLIANCE"
            
            # 2. Logic: Stress / Jitter
            current_pose_vector = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            
            if self.history:
                prev_vector = self.history[-1]
                diff = np.linalg.norm(current_pose_vector - prev_vector)
                
                # Check for nan/inf
                if not np.isfinite(diff): diff = 0.0
                
                self.jitter_scores.append(diff)
                if len(self.jitter_scores) > 60: # Increased smoothing window
                    self.jitter_scores.pop(0)
                
                avg_jitter = np.mean(self.jitter_scores) if self.jitter_scores else 0
                
                # Tuned Sensitivity:
                # 0.05 is a reasonable idle noise floor for 33 landmarks
                # Multiplier reduced to 2 to avoid instant 1.0 spikes
                stress_score = min(max((avg_jitter - 0.08) * 2, 0), 1.0)
                metrics["stress_level"] = stress_score
                
                if stress_score > 0.8: # Higher threshold for alert
                    metrics["status"] = "HIGH STRESS / AGITATION"
                    metrics["anomalies"].append("Erratic Movement")
            
            self.history.append(current_pose_vector)
            if len(self.history) > 60:
                self.history.pop(0)

            # 3. Draw Overlay (Manual if needed)
            if self.has_drawing_utils:
                # Need to convert landmarks list back to a proto-like object if using legacy drawing_utils
                # Or just draw manually. Legacy drawing utils expects a normalized landmark list proto.
                # It's hard to mix APIs. Let's draw manually for robustness.
                pass
                
            # Manual Drawing
            h, w, _ = annotated_image.shape
            # Draw connections (subset)
            connections = [
                (11, 12), (11, 13), (13, 15), # Left Arm
                (12, 14), (14, 16),           # Right Arm
                (11, 23), (12, 24), (23, 24)  # Torso
            ]
            
            # Points
            for i, lm in enumerate(landmarks):
                # Only draw upper body (0-24)
                if i > 24: continue
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_image, (cx, cy), 4, (0, 255, 255), -1)
                
            # Lines
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Text
            if metrics["surrender_detected"]:
                cv2.putText(annotated_image, "HANDS UP DETECTED", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
            cv2.putText(annotated_image, f"Stress: {metrics['stress_level']:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if metrics['stress_level'] > 0.6 else (0, 255, 0), 2)
                            
        return annotated_image, metrics
