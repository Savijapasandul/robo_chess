import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/train7/weights/best.pt")
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def order_points(pts):
    """Sort points in the order: top-left, top-right, bottom-right, bottom-left"""
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def draw_chess_grid(image, size=512, squares=8):
    """Draw 8x8 grid lines on the image"""
    step = size // squares
    grid = image.copy()
    for i in range(1, squares):
        cv2.line(grid, (i*step, 0), (i*step, size), (0, 255, 255), 2)  # vertical lines
        cv2.line(grid, (0, i*step), (size, i*step), (0, 255, 255), 2)  # horizontal lines
    return grid

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes
    names = results[0].names

    annotated_frame = frame.copy()
    found_chessboard = False

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]

        if label != "chessboard":
            continue

        found_chessboard = True

        # Draw polygon if available
        if hasattr(box, 'xy') and box.xy is not None:
            pts = box.xy[0].cpu().numpy().astype(int)
            if pts.shape[0] == 4:  # Expecting 4 corners
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                for (x, y) in pts:
                    cv2.circle(annotated_frame, (x, y), 6, (0, 0, 255), -1)

                # Warp perspective
                rect = order_points(pts)
                dst_size = 512
                dst = np.array([
                    [0, 0],
                    [dst_size - 1, 0],
                    [dst_size - 1, dst_size - 1],
                    [0, dst_size - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (dst_size, dst_size))
                warped_with_grid = draw_chess_grid(warped)
                cv2.imshow("Top-down Chessboard with Grid", warped_with_grid)
            else:
                print("Warning: Detected chessboard polygon doesn't have 4 points.")
        else:
            # Fallback to bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Detected Chessboard (Polygon)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        if found_chessboard:
            break

cap.release()
cv2.destroyAllWindows()
