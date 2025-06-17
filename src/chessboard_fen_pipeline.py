import cv2
import numpy as np
from ultralytics import YOLO

# Load segmentation model
model = YOLO("runs/segment/train/weights/best.pt")

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Cannot open webcam index 2")
    exit()

square_size = 100  # Warped board is 800x800, so each square is 100x100

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

# Files and ranks as per your corner references
files = ['h','g','f','e','d','c','b','a']  # left to right in warped image (cols)
ranks = ['8','7','6','5','4','3','2','1']  # top to bottom in warped image (rows)

def detect_piece(square_img):
    """
    Dummy piece detector placeholder.
    Replace with your actual model / logic.
    Returns '' for empty square.
    """
    # Example: always empty for now
    return ''

def board_to_fen(board):
    fen_rows = []
    for row in board:
        fen_row = ''
        empty_count = 0
        for sq in row:
            if sq == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += sq
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return '/'.join(fen_rows)

def process_warped_board(warped_img):
    board = []
    for r in range(8):  # rows
        row_pieces = []
        for c in range(8):  # cols
            x_start = c * square_size
            y_start = r * square_size
            square_img = warped_img[y_start:y_start + square_size, x_start:x_start + square_size]

            # For debugging, draw square borders
            cv2.rectangle(warped_img, (x_start, y_start), (x_start + square_size, y_start + square_size), (255,0,0), 1)

            piece = detect_piece(square_img)
            row_pieces.append(piece)
        board.append(row_pieces)

    # Your corner refs mean col 0= file h (rightmost), so reverse each row cols
    reordered_board = [row[::-1] for row in board]

    fen = board_to_fen(reordered_board)
    return fen, warped_img

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break

    results = model(frame)

    masks = results[0].masks
    if masks is None or len(masks.data) == 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

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

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        ordered_corners = order_points(pts)

        # Draw corners on frame
        for i, corner in enumerate(ordered_corners):
            cv2.circle(frame, tuple(corner.astype(int)), 10, (0,255,0), -1)
            cv2.putText(frame, f"{i}", tuple(corner.astype(int) + [10,-10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        width, height = 800, 800
        dst = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered_corners, dst)
        warped = cv2.warpPerspective(frame, M, (width, height))

        fen, board_visual = process_warped_board(warped)

        cv2.putText(frame, f"FEN: {fen}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Warped Board with Squares", board_visual)

    else:
        cv2.putText(frame, "4 corners not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
