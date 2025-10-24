import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import deque

with open('sign_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

valid_labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
PRED_WINDOW = 20
STABILITY_THRESHOLD = 15
pred_queue = deque(maxlen=PRED_WINDOW)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    pred = "-"
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        arr = np.array([[pt.x, pt.y, pt.z] for pt in lm]).flatten().reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred_raw = model.predict(arr_scaled)[0]
        pred = pred_raw if pred_raw in valid_labels else "-"
        mp.solutions.drawing_utils.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    pred_queue.append(pred)
    if pred != "-" and pred_queue.count(pred) >= STABILITY_THRESHOLD:
        stable_pred = pred
    else:
        stable_pred = "-"

    cv2.putText(frame, f"Sign: {stable_pred}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    cv2.imshow('Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
