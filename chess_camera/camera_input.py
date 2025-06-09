# === camera_input.py ===
import cv2

def get_camera_frame():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to read frame from camera")
    return frame