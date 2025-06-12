import cv2
import numpy as np
from ultralytics import YOLO

# === Parameters ===
MODEL_PATH = "runs/detect/train4/weights/best.pt"
CAMERA_INDEX = 0
GRID_SIZE = 8
LINE_COLOR = (0, 255, 255)
CORNER_COLOR = (0, 255, 0)
CORNER_RADIUS = 5
LINE_THICKNESS = 1

# === Load model ===
model = YOLO(MODEL_PATH)

# === Open webcam ===
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def get_box_corners(x1, y1, x2, y2):
    """Return 4 corner points (TL, TR, BR, BL)"""
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def draw_corners(frame, corners):
    """Draw green circles at each corner"""
    for i, (x, y) in enumerate(corners):
        cv2.circle(frame, (x, y), CORNER_RADIUS, CORNER_COLOR, -1)
        cv2.putText(frame, f"P{i+1}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORNER_COLOR, 1)

def draw_grid(frame, corners, rows=8, cols=8):
    """Draw grid lines inside the bounding box"""
    # Define grid lines by interpolating between corners
    top_left, top_right, bottom_right, bottom_left = corners
    for i in range(1, cols):
        alpha = i / cols
        x1 = int(top_left[0] * (1 - alpha) + top_right[0] * alpha)
        y1 = int(top_left[1] * (1 - alpha) + top_right[1] * alpha)
        x2 = int(bottom_left[0] * (1 - alpha) + bottom_right[0] * alpha)
        y2 = int(bottom_left[1] * (1 - alpha) + bottom_right[1] * alpha)
        cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)

    for j in range(1, rows):
        beta = j / rows
        x1 = int(top_left[0] * (1 - beta) + bottom_left[0] * beta)
        y1 = int(top_left[1] * (1 - beta) + bottom_left[1] * beta)
        x2 = int(top_right[0] * (1 - beta) + bottom_right[0] * beta)
        y2 = int(top_right[1] * (1 - beta) + bottom_right[1] * beta)
        cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]

        if name.lower() == "chessboard":
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            corners = get_box_corners(x1, y1, x2, y2)

            draw_corners(frame, corners)
            draw_grid(frame, corners)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Chessboard", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            break  # Only process one board

    cv2.imshow("Chessboard Detection with Grid", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
