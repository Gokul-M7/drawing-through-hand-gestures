import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define colors
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # Red, Green, Blue, Yellow
color_index = 0  # Default color

# Set up deque for smooth drawing
draw_points = [deque(maxlen=512) for _ in range(4)]

# Create a black canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# OpenCV Video Capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Added DirectShow backend for Windows

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the image
    h, w, c = frame.shape

    # Convert image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw the color selection buttons
    for i, col in enumerate(colors):
        cv2.rectangle(frame, (50 + i * 70, 10), (120 + i * 70, 60), col, -1)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index finger tip (landmark 8)
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * w), int(index_finger.y * h)

            # Check if selecting a color
            selected = False
            for i in range(len(colors)):
                if 50 + i * 70 < x < 120 + i * 70 and 10 < y < 60:
                    color_index = i
                    draw_points[color_index].clear()  # Reset instead of appending new deques
                    selected = True
                    break

            if not selected:
                draw_points[color_index].append((x, y))

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw the tracked points on the canvas
    for i, points in enumerate(draw_points):
        for j in range(1, len(points)):
            if isinstance(points[j - 1], tuple) and isinstance(points[j], tuple):
                cv2.line(canvas, points[j - 1], points[j], colors[i], 5)

    # Merge canvas with the webcam feed
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show the output
    cv2.imshow("AI Air Canvas", frame)

    # Clear drawing when 'c' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_points = [deque(maxlen=512) for _ in range(4)]  # Reset points

    # Exit when 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
