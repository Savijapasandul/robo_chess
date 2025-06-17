import cv2
import numpy as np
from ultralytics import YOLO
import chess
import time

# Load your trained segmentation model
model = YOLO("runs/segment/train/weights/best.pt")

# Open webcam index 2
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Cannot open webcam index 2")
    exit()

# Board settings
square_size = 100  # 800x800 warped board → 100x100 px squares
width, height = 800, 800

# For mapping to chess square names (based on your corner order: h8, h1, a1, a8)
files = ['h','g','f','e','d','c','b','a']
ranks = ['8','7','6','5','4','3','2','1']

# Initialize python-chess board
board = chess.Board()
prev_warped = None
last_move_time = 0

def order_points(pts):
    """Return ordered corners: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left (h8)
    rect[2] = pts[np.argmax(s)]    # bottom-right (a1)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right (h1)
    rect[3] = pts[np.argmax(diff)] # bottom-left (a8)
    return rect

def square_name(r, c):
    """Convert (row, col) to chess square notation based on warped image grid"""
    file = files[c]
    rank = ranks[r]
    return file + rank

def get_changed_squares(current_board, previous_board, threshold=30000):
    """Return list of squares that changed significantly between two board images"""
    changed = []
    for r in range(8):
        for c in range(8):
            x = c * square_size
            y = r * square_size
            curr = current_board[y:y+square_size, x:x+square_size]
            prev = previous_board[y:y+square_size, x:x+square_size]
            diff = cv2.absdiff(curr, prev)
            score = np.sum(diff)
            if score > threshold:
                changed.append((r, c))
    return changed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Segment the board using YOLO
    results = model(frame)
    masks = results[0].masks

    if masks is None or len(masks.data) == 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Extract the largest contour (chessboard)
    mask = masks.data[0].cpu().numpy().astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        cv2.putText(frame, "Chessboard corners not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Warp the board to top-down view (800x800)
    pts = approx.reshape(4, 2)
    ordered = order_points(pts)
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame, M, (width, height))

    # Visualize grid overlay on warped board
    warped_visual = warped.copy()
    for r in range(8):
        for c in range(8):
            x = c * square_size
            y = r * square_size
            cv2.rectangle(warped_visual, (x, y), (x + square_size, y + square_size), (255, 0, 0), 1)

    # Compare with previous warped board to detect moves
    if prev_warped is not None:
        changed = get_changed_squares(warped, prev_warped)

        # Only detect if 2 squares changed and delay between moves
        if len(changed) == 2 and time.time() - last_move_time > 1.5:
            (r1, c1), (r2, c2) = changed
            s1 = square_name(r1, c1)
            s2 = square_name(r2, c2)

            # Try both directions (source/dest ambiguity)
            for move_str in [s1+s2, s2+s1]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        print(f"✅ Legal move: {move_str}")
                        print("FEN:", board.fen())
                        last_move_time = time.time()
                        break
                except:
                    continue
        elif len(changed) > 2:
            print("⚠️ More than 2 squares changed — possible hand movement or drag")
        elif len(changed) == 1:
            print("⚠️ Only one square changed — unclear action")

    # Update previous warped image
    prev_warped = warped.copy()

    # Display visuals
    cv2.imshow("Live Chess Feed", frame)
    cv2.imshow("Warped Board View", warped_visual)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
