from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def detect_board(model, frame):
    results = model(frame)
    boxes = results[0].boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] == "chessboard":
            return box.xyxy[0].cpu().numpy().astype(int)
    return None
