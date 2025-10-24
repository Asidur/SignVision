import cv2
import mediapipe as mp
import numpy as np
import csv

labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]  # A–I, K–Y
filename = 'sign_data.csv'
samples_per_label = 200  # Change as needed

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
data = []

for label in labels:
    print(f"Show sign for {label}. Press 'c' to collect, 'q' to quit.")
    collected_count = 0
    collecting = False
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"Label: {label} Count: {collected_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collect Data', frame)
        key = cv2.waitKey(10)
        if key == ord('c'):
            collecting = True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
        if collecting and res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark
            row = []
            for lm in lms:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)
            data.append(row)
            collected_count += 1
            if collected_count >= samples_per_label:
                break

with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

cap.release()
cv2.destroyAllWindows()
print(f"Saved data to {filename}")
