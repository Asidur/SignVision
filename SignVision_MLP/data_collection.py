# Importing required Python libraries
import cv2  # For camera and image processing
import mediapipe as mp  # For hand landmark detection
import numpy as np  # For numerical operations
import os  # For folder and file management
import time  # For delays (not used much here)

# Directory where the collected data will be saved
DATA_DIR = './asl_dataset_fast'

# Number of images to collect per alphabet label
SAMPLES_PER_LABEL = 500

# All labels we want to capture (A-Z)
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands

# Configure the hand detection model
hands = mp_hands.Hands(
    static_image_mode=False,  # Track hands in real-time video
    max_num_hands=1,  # Only one hand allowed
    min_detection_confidence=0.5,  # Minimum confidence for detection
    min_tracking_confidence=0.5   # Tracking confidence
)

# Drawing utilities to visualize hand landmarks on screen
mp_drawing = mp.solutions.drawing_utils

# Open laptop camera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Loop through each alphabet label
for label in LABELS:

    # Create a folder for each label if it doesn't exist
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    print(f'Starting collection for letter: {label}')

    # Wait until user presses 's' to start capturing data for this label
    while True:
        ret, frame = cap.read()  # Read frame from webcam
        frame = cv2.flip(frame, 1)  # Mirror the image for natural view
        
        # Display instruction message on screen
        cv2.putText(frame, f'Show sign for "{label}". Press "s" to start capturing.',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Data Collection', frame)
        
        # Wait for key press, if 's' is pressed → break the waiting loop
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Start capturing samples for this label
    sample_count = 0
    while sample_count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()  # Capture camera frame
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB because MediaPipe uses RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hand landmarks
        results = hands.process(frame_rgb)

        # Copy frame for displaying drawings
        display_frame = frame.copy()

        # If a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw hand landmarks on display frame
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Collect standardized hand landmark coordinates
                landmarks_list = []

                # Reference wrist position to normalize values
                wrist = hand_landmarks.landmark[0]

                # Store relative (x, y) coordinates of each landmark
                for lm in hand_landmarks.landmark:
                    landmarks_list.append(lm.x - wrist.x)
                    landmarks_list.append(lm.y - wrist.y)

                # Save the landmark data as a .npy file
                file_path = os.path.join(label_dir, f'{sample_count}.npy')
                np.save(file_path, np.array(landmarks_list))

                sample_count += 1
                print(f'Saved sample {sample_count} for {label}')

        # Display progress text on screen
        cv2.putText(display_frame, f'Capturing for "{label}"', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)
        
        cv2.putText(display_frame, f'Samples: {sample_count}/{SAMPLES_PER_LABEL}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)

        # Show the live feed window
        cv2.imshow('Data Collection', display_frame)

        # Press 'q' anytime to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After finishing all data collection
print("Data collection complete!")

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
