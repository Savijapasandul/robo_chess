import cv2
import time
from ultralytics import YOLO

# Load the YOLOv11 model
# model = YOLO('/home/savija/projects/robo_chess/dataset/runs/detect/train2/weights/best.pt')
model = YOLO('best.pt')

# Open the webcam
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow('YOLOv11 Webcam Detection', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Delay before next detection (0.5 seconds)
    # time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()