import cv2
import numpy as np
from ultralytics import YOLO

# Load trained segmentation model
model = YOLO("runs/segment/train/weights/best.pt")

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def order_points(pts):
    """Sort polygon points to top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def draw_chess_grid(image, squares=8):
    """Draw 8x8 chess grid lines on an image"""
    h, w = image.shape[:2]
    step_x = w // squares
    step_y = h // squares
    grid = image.copy()

    for i in range(1, squares):
        # vertical lines
        cv2.line(grid, (i * step_x, 0), (i * step_x, h), (0, 255, 255), 2)
        # horizontal lines
        cv2.line(grid, (0, i * step_y), (w, i * step_y), (0, 255, 255), 2)
    return grid

locked_corners = None  # Will store the latest detected chessboard corners

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    masks = results[0].masks

    # On the live feed: draw segmentation masks if any
    if masks is not None and len(masks.xy) > 0:
        # Draw all detected masks on the frame (optional: with transparency)
        for seg_pts in masks.xy:
            pts = np.array(seg_pts, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(frame, [pts], color=(0, 255, 0, 50))

        # Attempt to find a chessboard mask with 4 corners and update locked corners continuously
        seg_pts = np.array(masks.xy[0], dtype=np.float32)
        epsilon = 0.02 * cv2.arcLength(seg_pts, True)
        approx = cv2.approxPolyDP(seg_pts, epsilon, True)

        if approx is not None and len(approx) == 4:
            locked_corners = order_points(approx[:, 0, :])
            # Convert approx points to int32 before drawing
            cv2.polylines(frame, [approx.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.putText(frame, "Chessboard Locked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # If corners detected, show warped top-down grid window, update live
    if locked_corners is not None:
        dst_size = 512
        dst_pts = np.array([
            [0, 0],
            [dst_size - 1, 0],
            [dst_size - 1, dst_size - 1],
            [0, dst_size - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(locked_corners, dst_pts)
        warped = cv2.warpPerspective(frame, M, (dst_size, dst_size))
        warped_grid = draw_chess_grid(warped)

        cv2.imshow("Top-down Chessboard Grid", warped_grid)
    else:
        # If no corners detected yet, clear the chessboard window or display message
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(blank, "No Chessboard Detected", (50, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Top-down Chessboard Grid", blank)

    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        print("[INFO] Resetting locked corners.")
        locked_corners = None  # Reset locked corners to allow fresh detection

cap.release()
cv2.destroyAllWindows()
