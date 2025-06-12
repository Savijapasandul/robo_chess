import cv2

def open_camera(index=2):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    return cap
