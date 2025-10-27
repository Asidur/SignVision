# Import all required libraries
import os
import cv2  # For video processing and camera access
import mediapipe as mp  # Hand tracking
import numpy as np
import tensorflow as tf  # For loading trained model
import tkinter as tk  # GUI library
from PIL import Image, ImageTk  # To convert OpenCV frames for Tkinter display


# ---------------------------
# 1️⃣ Load Model & Labels
# ---------------------------
MODEL_PATH = 'asl_recognizer_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)  # Load trained model into memory

# Dataset folder same as used during training
DATA_DIR = './asl_dataset_fast'

# Automatically read folder names (A-Z) and sort
LABELS = sorted(os.listdir(DATA_DIR))  


# ---------------------------
# 2️⃣ Initialize MediaPipe Hand Detection
# ---------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,  # Track real-time movement
    max_num_hands=1,  # Only one hand detection allowed at a time
    min_detection_confidence=0.7,  # Strict detection to avoid errors
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils  # For drawing landmark skeleton


# ---------------------------
# 3️⃣ Tkinter UI Application
# ---------------------------
class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Start webcam feed
        self.cap = cv2.VideoCapture(0)

        # Create video display area inside window
        self.canvas = tk.Canvas(
            window,
            width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # Create text pad area for detected letters (Notepad-like)
        notepad_frame = tk.Frame(window, padx=10, pady=10)
        notepad_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        notepad_label = tk.Label(notepad_frame, text="Recognized Text:", font=("Arial", 14))
        notepad_label.pack(anchor=tk.W)

        self.notepad = tk.Text(notepad_frame, height=20, width=40, font=("Arial", 16))
        self.notepad.pack()

        # Prediction smoothing variables
        self.prediction_history = []  # Stores most recent predictions
        self.last_char = ""  # Prevent repeated letters like AAAAA

        # Start the main update loop
        self.update()
        self.window.mainloop()

    def update(self):
        """Main loop: Runs continuously to update video and prediction."""
        ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror for natural feel

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            display_frame = frame.copy()  # Copy for rendering

            if results.multi_hand_landmarks:
                # If hand detected → draw and predict
                for hand_landmarks in results.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    # Create feature vector (42 numbers = 21 x,y coords)
                    landmarks_list = []
                    wrist = hand_landmarks.landmark[0]

                    for lm in hand_landmarks.landmark:
                        landmarks_list.append(lm.x - wrist.x)
                        landmarks_list.append(lm.y - wrist.y)

                    # Model prediction (returns confidence for each class)
                    prediction = model.predict(
                        np.expand_dims(landmarks_list, axis=0),
                        verbose=0
                    )
                    predicted_class_index = np.argmax(prediction)
                    confidence = np.max(prediction)

                    # Show detected letter if confidence high enough
                    if confidence > 0.90:  
                        predicted_char = LABELS[predicted_class_index]

                        # Dark box behind text for visibility
                        cv2.rectangle(display_frame, (10, 10), (180, 70), (0, 0, 0), -1)
                        cv2.putText(display_frame, f'{predicted_char}',
                                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 255, 255), 3, cv2.LINE_AA)

                        # Smooth prediction → Update notepad text
                        self.update_notepad(predicted_char)

            # Convert for Tkinter display
            self.photo = ImageTk.PhotoImage(
                image=Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            )
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Run again after 10ms (simulates FPS)
        self.window.after(10, self.update)

    def update_notepad(self, char):
        """Update text only when stable prediction is achieved."""
        self.prediction_history.append(char)

        # Only keep latest 10 predictions
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)

        # Check stability → All predictions are same
        is_stable = len(set(self.prediction_history)) == 1

        # Prevent duplicate insertion of same letter
        if is_stable and self.prediction_history[0] != self.last_char:
            self.notepad.insert(tk.END, self.prediction_history[0])
            self.last_char = self.prediction_history[0]

    def __del__(self):
        # Properly release webcam when app closes
        if self.cap.isOpened():
            self.cap.release()


# ---------------------------
# 4️⃣ Start the App
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root, "ASL Sign Language Recognizer")
