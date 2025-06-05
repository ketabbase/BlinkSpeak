import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp
import sys
import os

class ProfessionalEyeTracker:
    def __init__(self, webcam_id=0):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.webcam = cv2.VideoCapture(webcam_id)
        if not self.webcam.isOpened():
            raise IOError("Could not open webcam. Please check your webcam connection and drivers.")
        
        # Get webcam properties
        self.frame_width = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.webcam.get(cv2.CAP_PROP_FPS)
        
        # Initialize data storage
        self.is_recording = False
        self.start_time = None
        
        # Define iris landmarks indices (MediaPipe's iris landmarks)
        self.LEFT_IRIS = [474, 475, 476, 477]  # Left eye iris landmarks
        self.RIGHT_IRIS = [469, 470, 471, 472]  # Right eye iris landmarks
        
        # Define eye corner landmarks for reference
        self.LEFT_EYE_CORNERS = [33, 133]  # Left eye corners
        self.RIGHT_EYE_CORNERS = [362, 263]  # Right eye corners

        # Define inner lip border landmarks (where upper and lower lips meet)
        self.INNER_LIP_BORDER = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 87
        ]

        # Window size tracking
        self.window_width = self.frame_width
        self.window_height = self.frame_height
        self.scale_factor = 1.0
        self.display_scale = 1.0

        # Blink tracking
        self.eye_state = "open"  # Current state: "open" or "closed"
        self.blink_start_time = None
        self.blinks = []
        self.eyes_closed_threshold = 0.05  # Eyes are closed when height is less than 5% of width
        self.eyes_open_threshold = 0.1    # Eyes are open when height is more than 10% of width
        self.last_eye_heights = []         # Store recent eye heights for stability
        self.history_size = 3              # Number of frames to keep in history
        self.last_blink_time = 0          # Track time of last blink
        self.min_blink_interval = 0.1     # Minimum time between blinks (seconds)

        # Mouth tracking
        self.mouth_state = "closed"  # Current state: "open" or "closed"
        self.mouth_start_time = None
        self.mouth_movements = []
        self.mouth_closed_threshold = 0.1  # Mouth is closed when height is less than 10% of width
        self.mouth_open_threshold = 0.2   # Mouth is open when height is more than 20% of width
        self.last_mouth_heights = []      # Store recent mouth heights for stability
        self.last_mouth_time = 0         # Track time of last mouth movement

        # Speech detection
        self.speech_state = "not_speaking"  # Current state: "speaking" or "not_speaking"
        self.speech_start_time = None
        self.speech_segments = []
        self.mouth_movement_history = []  # Store recent mouth movements for analysis
        self.speech_window = 1.0  # Time window to analyze for speech (seconds)
        self.min_mouth_movements = 3  # Minimum number of mouth movements in window to consider as speech
        self.min_movement_duration = 0.05  # Minimum duration of a mouth movement to count
        self.max_movement_duration = 0.5  # Maximum duration of a mouth movement to count
        self.last_speech_time = 0  # Track time of last speech detection
        self.speech_cooldown = 0.2  # Time to wait before detecting new speech segment
        self.last_mouth_state = "closed"  # Track previous mouth state for transition detection

    def resize_frame(self, frame, target_width=None, target_height=None):
        """Resize frame while maintaining aspect ratio."""
        if target_width is None and target_height is None:
            return frame

        # Get screen dimensions
        screen_width = cv2.getWindowImageRect("Eye Tracking System")[2]
        screen_height = cv2.getWindowImageRect("Eye Tracking System")[3]

        # Calculate maximum dimensions that maintain aspect ratio
        aspect_ratio = self.frame_width / self.frame_height
        
        if target_width is None:
            target_width = int(target_height * aspect_ratio)
        elif target_height is None:
            target_height = int(target_width / aspect_ratio)

        # Ensure the frame fits within the screen
        if target_width > screen_width:
            target_width = screen_width
            target_height = int(target_width / aspect_ratio)
        if target_height > screen_height:
            target_height = screen_height
            target_width = int(target_height * aspect_ratio)

        self.window_width = target_width
        self.window_height = target_height
        self.scale_factor = target_width / self.frame_width
        self.display_scale = min(screen_width / self.frame_width, screen_height / self.frame_height)

        return cv2.resize(frame, (target_width, target_height))

    def get_eye_height(self, landmarks, eye_indices):
        """Calculate eye height based on upper and lower eyelid landmarks."""
        # Get upper and lower eyelid points
        upper_points = [(landmarks.landmark[idx].x * self.frame_width,
                        landmarks.landmark[idx].y * self.frame_height)
                       for idx in eye_indices[:8]]  # Upper eyelid points
        lower_points = [(landmarks.landmark[idx].x * self.frame_width,
                        landmarks.landmark[idx].y * self.frame_height)
                       for idx in eye_indices[8:]]  # Lower eyelid points
        
        # Calculate average y-coordinate for upper and lower eyelids
        upper_y = np.mean([p[1] for p in upper_points])
        lower_y = np.mean([p[1] for p in lower_points])
        
        # Calculate eye width using the outermost points
        left_x = min(p[0] for p in upper_points + lower_points)
        right_x = max(p[0] for p in upper_points + lower_points)
        eye_width = right_x - left_x
        
        # Calculate eye height relative to eye width
        eye_height = abs(lower_y - upper_y) / eye_width
        
        return eye_height

    def get_mouth_height(self, landmarks):
        """Calculate mouth height based on inner lip border landmarks."""
        # Get inner lip border points
        lip_points = [(landmarks.landmark[idx].x * self.frame_width,
                      landmarks.landmark[idx].y * self.frame_height)
                     for idx in self.INNER_LIP_BORDER]
        
        # Calculate mouth width and height
        left_x = min(p[0] for p in lip_points)
        right_x = max(p[0] for p in lip_points)
        top_y = min(p[1] for p in lip_points)
        bottom_y = max(p[1] for p in lip_points)
        
        mouth_width = right_x - left_x
        mouth_height = bottom_y - top_y
        
        # Calculate relative height
        relative_height = mouth_height / mouth_width
        
        return relative_height

    def detect_blink(self, current_eye_height, current_time, frame):
        """Simple blink detection using relative eye height."""
        # Update eye height history
        self.last_eye_heights.append(current_eye_height)
        if len(self.last_eye_heights) > self.history_size:
            self.last_eye_heights.pop(0)
        
        # Use average of recent heights for more stable detection
        avg_eye_height = np.mean(self.last_eye_heights)
        
        # Debug print for eye height
        if self.is_recording:
            cv2.putText(frame, f"Eye Height: {avg_eye_height:.2f}",
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Only detect blinks when recording
        if not self.is_recording:
            return
        
        # Check if enough time has passed since last blink
        time_since_last_blink = current_time - self.last_blink_time
        
        # State machine for blink detection
        if self.eye_state == "open":
            # Check for transition to closed (height close to 0)
            if avg_eye_height < self.eyes_closed_threshold and time_since_last_blink > self.min_blink_interval:
                self.eye_state = "closed"
                self.blink_start_time = current_time
                print(f"Blink started at {time.strftime('%H:%M:%S', time.localtime(current_time))}.{int((current_time % 1) * 1000000)}")
                cv2.putText(frame, "Eyes Closed",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        elif self.eye_state == "closed":
            # Check for transition to open (height becomes non-zero)
            if avg_eye_height > self.eyes_open_threshold:
                self.eye_state = "open"
                blink_duration = current_time - self.blink_start_time
                self.last_blink_time = current_time
                
                # Record the blink
                relative_time = current_time - self.start_time
                self.blinks.append({
                    'start_time': f"{time.strftime('%H:%M:%S', time.localtime(self.blink_start_time))}.{int((self.blink_start_time % 1) * 1000000)}",
                    'end_time': f"{time.strftime('%H:%M:%S', time.localtime(current_time))}.{int((current_time % 1) * 1000000)}",
                    'duration': blink_duration,
                    'relative_time': relative_time
                })
                print(f"Blink ended at {time.strftime('%H:%M:%S', time.localtime(current_time))}.{int((current_time % 1) * 1000000)}, duration: {blink_duration:.2f}s")
                
                # Draw blink information
                cv2.putText(frame, f"Last Blink: {blink_duration:.1f}s",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Time: {relative_time:.1f}s",
                          (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Update display based on current state
        if self.eye_state == "closed":
            current_duration = current_time - self.blink_start_time
            cv2.putText(frame, f"Eyes Closed: {current_duration:.1f}s",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "Eyes Open",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Force reset to open state if eyes are clearly open
        if avg_eye_height > self.eyes_open_threshold * 1.2:  # Add some margin
            self.eye_state = "open"

    def detect_mouth_movement(self, current_mouth_height, current_time, frame):
        """Detect mouth movements using relative mouth height."""
        # Update mouth height history
        self.last_mouth_heights.append(current_mouth_height)
        if len(self.last_mouth_heights) > self.history_size:
            self.last_mouth_heights.pop(0)
        
        # Use average of recent heights for more stable detection
        avg_mouth_height = np.mean(self.last_mouth_heights)
        
        # Debug print for mouth height
        if self.is_recording:
            cv2.putText(frame, f"Mouth Height: {avg_mouth_height:.2f}",
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Only detect mouth movements when recording
        if not self.is_recording:
            return
        
        # Check if enough time has passed since last movement
        time_since_last_movement = current_time - self.last_mouth_time
        
        # State machine for mouth movement detection
        if self.mouth_state == "closed":
            # Check for transition to open
            if avg_mouth_height > self.mouth_open_threshold and time_since_last_movement > self.min_blink_interval:
                self.mouth_state = "open"
                self.mouth_start_time = current_time
                print(f"Mouth opened at {time.strftime('%H:%M:%S', time.localtime(current_time))}.{int((current_time % 1) * 1000000)}")
                cv2.putText(frame, "Mouth Open",
                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        elif self.mouth_state == "open":
            # Check for transition to closed
            if avg_mouth_height < self.mouth_closed_threshold:
                self.mouth_state = "closed"
                movement_duration = current_time - self.mouth_start_time
                self.last_mouth_time = current_time
                
                # Record the mouth movement
                relative_time = current_time - self.start_time
                self.mouth_movements.append({
                    'start_time': self.mouth_start_time,
                    'end_time': current_time,
                    'duration': movement_duration,
                    'relative_time': relative_time,
                    'transition': f"{self.last_mouth_state}->{self.mouth_state}"
                })
                print(f"Mouth closed at {time.strftime('%H:%M:%S', time.localtime(current_time))}.{int((current_time % 1) * 1000000)}, duration: {movement_duration:.2f}s")
                
                # Draw mouth movement information
                cv2.putText(frame, f"Last Movement: {movement_duration:.1f}s",
                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Update display based on current state
        if self.mouth_state == "open":
            current_duration = current_time - self.mouth_start_time
            cv2.putText(frame, f"Mouth Open: {current_duration:.1f}s",
                      (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "Mouth Closed",
                      (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def detect_speech(self, current_time, frame):
        """Detect if the user is speaking based on mouth movement transitions."""
        # Only detect speech when recording
        if not self.is_recording:
            return

        # Check for mouth state transitions
        if self.mouth_state != self.last_mouth_state:
            # Record the transition
            self.mouth_movements.append({
                'start_time': current_time,
                'end_time': current_time,
                'duration': 0.0,  # Will be updated on next transition
                'relative_time': current_time - self.start_time,
                'transition': f"{self.last_mouth_state}->{self.mouth_state}"
            })
            
            # Update previous movement's duration if it exists
            if len(self.mouth_movements) > 1:
                prev_movement = self.mouth_movements[-2]
                prev_movement['duration'] = current_time - prev_movement['start_time']
                prev_movement['end_time'] = current_time

        # Debug information
        if self.is_recording:
            cv2.putText(frame, f"Last transition: {self.last_mouth_state}->{self.mouth_state}",
                      (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Update display based on mouth movement
        if self.mouth_state != self.last_mouth_state:
            cv2.putText(frame, "Speaking",
                      (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Update last mouth state
        self.last_mouth_state = self.mouth_state

    def process_frame(self):
        ret, frame = self.webcam.read()
        if not ret:
            return None

        # Store current frame for debugging
        self.current_frame = frame.copy()

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate eye heights
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            left_eye_height = self.get_eye_height(face_landmarks, left_eye_indices)
            right_eye_height = self.get_eye_height(face_landmarks, right_eye_indices)
            current_eye_height = min(left_eye_height, right_eye_height)  # Use minimum height for more accurate blink detection
            
            # Calculate mouth height
            current_mouth_height = self.get_mouth_height(face_landmarks)
            
            # Enhanced blink detection
            current_time = time.time()
            self.detect_blink(current_eye_height, current_time, frame)
            self.detect_mouth_movement(current_mouth_height, current_time, frame)
            self.detect_speech(current_time, frame)
            
            # Draw eyelids
            # Left eye
            left_eye_points = []
            for idx in left_eye_indices:
                x = int(face_landmarks.landmark[idx].x * self.frame_width * self.scale_factor)
                y = int(face_landmarks.landmark[idx].y * self.frame_height * self.scale_factor)
                left_eye_points.append((x, y))
            cv2.polylines(frame, [np.array(left_eye_points)], True, (0, 255, 0), 1)
            
            # Right eye
            right_eye_points = []
            for idx in right_eye_indices:
                x = int(face_landmarks.landmark[idx].x * self.frame_width * self.scale_factor)
                y = int(face_landmarks.landmark[idx].y * self.frame_height * self.scale_factor)
                right_eye_points.append((x, y))
            cv2.polylines(frame, [np.array(right_eye_points)], True, (0, 255, 0), 1)
            
            # Draw inner lip border
            lip_points = []
            for idx in self.INNER_LIP_BORDER:
                x = int(face_landmarks.landmark[idx].x * self.frame_width * self.scale_factor)
                y = int(face_landmarks.landmark[idx].y * self.frame_height * self.scale_factor)
                lip_points.append((x, y))
            cv2.polylines(frame, [np.array(lip_points)], True, (0, 255, 0), 1)  # Green for inner lip border

        # Add status text with proper scaling
        text = "Recording..." if self.is_recording else "Paused"
        font_scale = max(0.5, 1.0 * self.display_scale)
        font_thickness = max(1, int(2 * self.display_scale))
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Ensure text stays within visible area
        text_x = max(0, min(50, self.window_width - text_size[0] - 10))
        text_y = max(text_size[1] + 10, 50)
        
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

        return frame

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.blinks = []
            self.mouth_movements = []
            self.speech_segments = []
            self.start_time = time.time()
            print("Recording started...")
            print(f"Start time: {self.start_time:.2f}")

    def stop_recording(self):
        if self.is_recording:
            # If we're in the middle of a blink, complete it
            if self.eye_state == "closed":
                current_time = time.time()
                blink_duration = current_time - self.blink_start_time
                relative_time = current_time - self.start_time
                
                self.blinks.append({
                    'start_time': self.blink_start_time,
                    'end_time': current_time,
                    'duration': blink_duration,
                    'relative_time': relative_time
                })
                print(f"Completed final blink at {time.strftime('%H:%M:%S', time.localtime(current_time))}.{int((current_time % 1) * 1000000)}, duration: {blink_duration:.2f}s")
            
            # If we're in the middle of a mouth movement, complete it
            if self.mouth_state == "open":
                current_time = time.time()
                movement_duration = current_time - self.mouth_start_time
                relative_time = current_time - self.start_time
                
                self.mouth_movements.append({
                    'start_time': self.mouth_start_time,
                    'end_time': current_time,
                    'duration': movement_duration,
                    'relative_time': relative_time,
                    'transition': f"{self.last_mouth_state}->{self.mouth_state}"
                })
                print(f"Completed final mouth movement at {time.strftime('%H:%M:%S', time.localtime(current_time))}.{int((current_time % 1) * 1000000)}, duration: {movement_duration:.2f}s")
            
            # If we're in the middle of speech, complete it
            if self.speech_state == "speaking":
                current_time = time.time()
                speech_duration = current_time - self.speech_start_time
                relative_time = current_time - self.start_time
                
                self.speech_segments.append({
                    'start_time': self.speech_start_time,
                    'end_time': current_time,
                    'duration': speech_duration,
                    'relative_time': relative_time,
                    'mouth_movements': len([m for m in self.mouth_movements 
                                          if self.min_movement_duration <= m['duration'] <= self.max_movement_duration])
                })
                print(f"Completed final speech segment at {time.strftime('%H:%M:%S', time.localtime(current_time))}.{int((current_time % 1) * 1000000)}, duration: {speech_duration:.2f}s")
            
            self.is_recording = False
            self.eye_state = "open"  # Reset eye state
            self.mouth_state = "closed"  # Reset mouth state
            self.speech_state = "not_speaking"  # Reset speech state
            print("Recording stopped.")
            print(f"Total blinks detected: {len(self.blinks)}")
            print(f"Total mouth movements detected: {len(self.mouth_movements)}")
            print(f"Total speech segments detected: {len(self.speech_segments)}")
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)

            # Generate a timestamp for the filename
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")

            # Save blinks data if there are any blinks
            if self.blinks:
                # Convert timestamps to formatted strings for CSV
                blinks_data = []
                for blink in self.blinks:
                    blink_data = blink.copy()
                    if isinstance(blink['start_time'], (int, float)):
                        blink_data['start_time'] = f"{time.strftime('%H:%M:%S', time.localtime(blink['start_time']))}.{int((blink['start_time'] % 1) * 1000000)}"
                        blink_data['end_time'] = f"{time.strftime('%H:%M:%S', time.localtime(blink['end_time']))}.{int((blink['end_time'] % 1) * 1000000)}"
                    blinks_data.append(blink_data)
                
                blinks_df = pd.DataFrame(blinks_data)
                blinks_df.to_csv(f"data/{timestamp_str}-blinks.csv", index=False)
                print(f"Saved {len(self.blinks)} blinks to data/{timestamp_str}-blinks.csv")
                
                # Print blink statistics
                blink_stats = {
                    'total_closures': len(blinks_df),
                    'avg_duration': blinks_df['duration'].mean() if not blinks_df.empty else 0.0,
                    'min_duration': blinks_df['duration'].min() if not blinks_df.empty else 0.0,
                    'max_duration': blinks_df['duration'].max() if not blinks_df.empty else 0.0
                }
                
                print("\nEye Closure Statistics:")
                print(f"Total closures: {blink_stats['total_closures']}")
                print(f"Average duration: {blink_stats['avg_duration']:.1f}s")
                print(f"Min duration: {blink_stats['min_duration']:.1f}s")
                print(f"Max duration: {blink_stats['max_duration']:.1f}s")
            else:
                print("No blinks recorded.")
            
            # Save mouth movements data if there are any movements
            if self.mouth_movements:
                # Convert timestamps to formatted strings for CSV
                movements_data = []
                for movement in self.mouth_movements:
                    movement_data = movement.copy()
                    if isinstance(movement['start_time'], (int, float)):
                        movement_data['start_time'] = f"{time.strftime('%H:%M:%S', time.localtime(movement['start_time']))}.{int((movement['start_time'] % 1) * 1000000)}"
                        movement_data['end_time'] = f"{time.strftime('%H:%M:%S', time.localtime(movement['end_time']))}.{int((movement['end_time'] % 1) * 1000000)}"
                    movements_data.append(movement_data)
                
                mouth_movements_df = pd.DataFrame(movements_data)
                mouth_movements_df.to_csv(f"data/{timestamp_str}-mouth_movements.csv", index=False)
                print(f"Saved {len(self.mouth_movements)} mouth movements to data/{timestamp_str}-mouth_movements.csv")
                
                # Print mouth movement statistics
                mouth_stats = {
                    'total_movements': len(mouth_movements_df),
                    'avg_duration': mouth_movements_df['duration'].mean() if not mouth_movements_df.empty else 0.0,
                    'min_duration': mouth_movements_df['duration'].min() if not mouth_movements_df.empty else 0.0,
                    'max_duration': mouth_movements_df['duration'].max() if not mouth_movements_df.empty else 0.0
                }
                
                print("\nMouth Movement Statistics:")
                print(f"Total movements: {mouth_stats['total_movements']}")
                print(f"Average duration: {mouth_stats['avg_duration']:.1f}s")
                print(f"Min duration: {mouth_stats['min_duration']:.1f}s")
                print(f"Max duration: {mouth_stats['max_duration']:.1f}s")
            else:
                print("No mouth movements recorded.")
            
            # Save speech segments data if there are any segments
            if self.speech_segments:
                # Convert timestamps to formatted strings for CSV
                segments_data = []
                for segment in self.speech_segments:
                    segment_data = segment.copy()
                    if isinstance(segment['start_time'], (int, float)):
                        segment_data['start_time'] = f"{time.strftime('%H:%M:%S', time.localtime(segment['start_time']))}.{int((segment['start_time'] % 1) * 1000000)}"
                        segment_data['end_time'] = f"{time.strftime('%H:%M:%S', time.localtime(segment['end_time']))}.{int((segment['end_time'] % 1) * 1000000)}"
                    segments_data.append(segment_data)
                
                speech_df = pd.DataFrame(segments_data)
                speech_df.to_csv(f"data/{timestamp_str}-speech_segments.csv", index=False)
                print(f"Saved {len(self.speech_segments)} speech segments to data/{timestamp_str}-speech_segments.csv")
                
                # Print speech statistics
                speech_stats = {
                    'total_segments': len(speech_df),
                    'avg_duration': speech_df['duration'].mean() if not speech_df.empty else 0.0,
                    'min_duration': speech_df['duration'].min() if not speech_df.empty else 0.0,
                    'max_duration': speech_df['duration'].max() if not speech_df.empty else 0.0,
                    'total_speaking_time': speech_df['duration'].sum() if not speech_df.empty else 0.0,
                    'avg_movements_per_segment': speech_df['mouth_movements'].mean() if not speech_df.empty else 0.0
                }
                
                print("\nSpeech Statistics:")
                print(f"Total speech segments: {speech_stats['total_segments']}")
                print(f"Total speaking time: {speech_stats['total_speaking_time']:.1f}s")
                print(f"Average segment duration: {speech_stats['avg_duration']:.1f}s")
                print(f"Min segment duration: {speech_stats['min_duration']:.1f}s")
                print(f"Max segment duration: {speech_stats['max_duration']:.1f}s")
                print(f"Average mouth movements per segment: {speech_stats['avg_movements_per_segment']:.1f}")
            else:
                print("No speech segments recorded.")

    def release(self):
        """Releases the webcam resources."""
        if self.webcam:
            self.webcam.release()
        if self.face_mesh:
            self.face_mesh.close()
        print("System resources released.")

# --- Example Usage ---
if __name__ == "__main__":
    try:
        tracker = ProfessionalEyeTracker(webcam_id=0)
    except (IOError, RuntimeError) as e:
        print(f"Failed to initialize EyeTracker system: {e}")
        print("Please ensure your webcam is available.")
        sys.exit(1)

    print("Press 'R' to start recording.")
    print("Press 'S' to stop recording.")
    print("Press 'Q' to quit.")

    # Create a resizable window with initial webcam size
    cv2.namedWindow("Eye Tracking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye Tracking System", tracker.frame_width, tracker.frame_height)

    while True:
        frame = tracker.process_frame()
        if frame is None:
            break

        cv2.imshow("Eye Tracking System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            tracker.start_recording()
        elif key == ord('s'):
            tracker.stop_recording()
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    tracker.release()
    print("System finished.")
