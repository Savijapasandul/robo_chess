from ultralytics import YOLO
import cv2

model = YOLO('/home/savija/projects/robo_chess/runs/detect/train/weights/last.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    results = model(frame, show=True)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = model.names[class_id]
            confidence = float(box.conf[0])
            print(f"Detected {name} with {confidence:.2f} confidence")

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
