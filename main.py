import cv2
import mediapipe as mp
import numpy as np
import time
import onnxruntime as ort

# Detect GPU availability for ONNXRuntime
def has_gpu():
    providers = ort.get_available_providers()
    return 'CUDAExecutionProvider' in providers

use_gpu = has_gpu()
print(f"Using {'GPU' if use_gpu else 'CPU'} for inference")

# Initialize MediaPipe Hands (CPU-based)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam not detected.")
    exit(1)

last_positions = {}
threshold = 80
history_length = 20

def is_waving(hand_id, cx):
    if hand_id not in last_positions:
        last_positions[hand_id] = []
    last_positions[hand_id].append(cx)
    if len(last_positions[hand_id]) > history_length:
        last_positions[hand_id].pop(0)
    positions = last_positions[hand_id]
    if len(positions) >= 10:
        return np.ptp(positions) > threshold
    return False

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            cx = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
            cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

            if is_waving(i, cx):
                cv2.putText(frame, f"👋 Hand {i+1} Waving!", (50, 80 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Wave Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()