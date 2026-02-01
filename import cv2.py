import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Enable TensorFlow to manage GPU resources
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Curl counter variables
counter = 0
stage = None
goal_reps = 10  # Define your goal for reps

# Initialize mediapipe pose detection and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture image")
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make pose detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks and perform calculations
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for LEFT Arm (Curl counter)
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate the angle for curl counting
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == 'down':
                stage = "up"
                counter += 1
                print(f"Rep Count: {counter}")
            
            # Check if the dumbbell (wrist) is too close to the chest (shoulder)
            distance_threshold = 0.1  # Set an appropriate distance threshold for the warning
            distance = calculate_distance(left_wrist, left_shoulder)
            dumbbell_too_close = distance < distance_threshold

            # Posture Check
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            posture_correct = (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y and
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y < landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y and
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y < landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            )

            # Enhanced Feedback System - Green if within correct range, Red if outside
            angle_correct = angle < 160
            indicator_color = (0, 255, 0) if angle_correct else (0, 0, 255)  # Green for correct, Red for incorrect

            # Draw indicator circle
            cv2.circle(image, (600, 50), 20, indicator_color, -1)

            # Visualize the angle
            cv2.putText(image, str(int(angle)), 
                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Visualize alert if dumbbell is too close
            if dumbbell_too_close:
                cv2.putText(image, 'Too Close!', (250, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

            # Visualize curl counter and stage
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw progress bar
            bar_width = 300  # Width of the progress bar
            bar_height = 30  # Height of the progress bar
            progress = int((counter / goal_reps) * bar_width)  # Calculate progress based on reps
            cv2.rectangle(image, (10, 80), (10 + bar_width, 80 + bar_height), (255, 255, 255), -1)  # Background bar
            cv2.rectangle(image, (10, 80), (10 + progress, 80 + bar_height), (0, 255, 0), -1)  # Filled bar

            # Visual indicators for alignment
            cv2.line(image, tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                     tuple(np.multiply(left_hip, [640, 480]).astype(int)), (0, 255, 0) if posture_correct else (0, 0, 255), 2)

            # Render pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Display the output
        cv2.imshow('Posture and Curl Analysis', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()