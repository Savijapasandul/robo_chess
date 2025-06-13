import cv2
import numpy as np
from ultralytics import YOLO
import time

model = YOLO("runs/segment/train/weights/best.pt")
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

def order_points(pts):
    """Order points as top-left (a1), top-right (a8), bottom-right (h8), bottom-left (h1)"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left (a1)
    rect[2] = pts[np.argmax(s)]  # bottom-right (h8)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right (a8)
    rect[3] = pts[np.argmax(diff)]  # bottom-left (h1)
    return rect

def draw_chess_cells(warped_img, squares=8):
    h, w = warped_img.shape[:2]
    step_x = w / squares
    step_y = h / squares
    overlay = warped_img.copy()

    for row in range(squares):
        for col in range(squares):
            x1 = int(col * step_x)
            y1 = int(row * step_y)
            x2 = int((col + 1) * step_x)
            y2 = int((row + 1) * step_y)

            # Draw rectangle border (light blue)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 1)

            # Label each cell (e.g. a1, a2... h8) in small white font, top-left corner
            col_label = chr(ord('a') + col)  # a-h
            row_label = str(row + 1)          # 1-8 (top to bottom)
            label = f"{col_label}{row_label}"
            font_scale = 0.4
            thickness = 1
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = x1 + 3
            text_y = y1 + text_size[1] + 3
            cv2.putText(overlay, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, warped_img, 1 - alpha, 0, warped_img)
    return warped_img

locked_corners = None
last_update_time = 0
top_down_image = None
last_corners_for_display = None  # To store last corners for drawing between updates

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    masks = results[0].masks

    current_time = time.time()
    updated_corners = False

    if masks is not None and len(masks.xy) > 0:
        seg_pts = np.array(masks.xy[0], dtype=np.float32)
        epsilon = 0.02 * cv2.arcLength(seg_pts, True)
        approx = cv2.approxPolyDP(seg_pts, epsilon, True)

        if approx is not None and len(approx) == 4:
            locked_corners_candidate = order_points(approx[:, 0, :])

            # Update corners only if 5 seconds passed since last update
            if current_time - last_update_time > 5 or locked_corners is None:
                locked_corners = locked_corners_candidate
                last_update_time = current_time
                updated_corners = True

    # Draw corner markers and labels only every 5 seconds
    if locked_corners is not None:
        # Save current corners for display
        if updated_corners or last_corners_for_display is None:
            last_corners_for_display = locked_corners.copy()

        # Draw smaller red circles and smaller font labels p1-p4
        for i, point in enumerate(last_corners_for_display):
            pt = tuple(point.astype(int))
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)  # smaller radius=5
            cv2.putText(frame, f"p{i+1}", (pt[0] + 3, pt[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    if locked_corners is not None:
        # Update top-down view only if 5 seconds have passed or first time
        if updated_corners or top_down_image is None:
            dst_size = 512
            dst_pts = np.array([
                [0, 0],                   # a1 top-left
                [dst_size - 1, 0],        # a8 top-right
                [dst_size - 1, dst_size - 1],  # h8 bottom-right
                [0, dst_size - 1]         # h1 bottom-left
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(locked_corners, dst_pts)
            warped = cv2.warpPerspective(frame, M, (dst_size, dst_size))
            top_down_image = draw_chess_cells(warped)

        cv2.imshow("Top-down Chessboard Grid", top_down_image)
    else:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(blank, "No Chessboard Detected", (50, 256),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Top-down Chessboard Grid", blank)

    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        print("[INFO] Resetting locked corners.")
        locked_corners = None
        last_corners_for_display = None
        top_down_image = None
        last_update_time = 0

cap.release()
cv2.destroyAllWindows()
