# === piece_detection.py ===
from ultralytics import YOLO

# Load your trained YOLOv8 model
# model = YOLO('/home/savija/projects/robo_chess/dataset/runs/detect/train3/weights/best.pt')

model = YOLO('/home/savija/projects/robo_chess/runs/detect/train3/weights/best.pt')

# Ensure the model class names match yours
CLASS_NAMES = [
    'black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
    'chessboard',
    'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'
]

def detect_pieces(warped_img):
    results = model(warped_img)
    detections = []

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]

            # Skip the chessboard itself
            if class_name == "chessboard":
                continue

            x_center = float(box.xywh[0][0])
            y_center = float(box.xywh[0][1])

            detections.append((class_name, x_center, y_center))
    
    return detections