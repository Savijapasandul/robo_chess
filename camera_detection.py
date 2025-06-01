import cv2

# Replace this with your actual YOLOv11 model loading and inference code
class YOLOv11:
    def __init__(self, model_path):
        # Load your YOLOv11 model here
        pass

    def detect(self, frame):
        # Run detection and return results
        # Example: return [{'bbox': [x, y, w, h], 'confidence': 0.9, 'class': 'person'}]
        return []

model = YOLOv11('yolo11n.pt')

cap = cv2.VideoCapture(2)  # USB webcam index 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = model.detect(frame)

    # Draw detections
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv11 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()