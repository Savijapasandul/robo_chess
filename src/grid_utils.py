import cv2

GRID_SIZE = 8
WARP_SIZE = 400
SQUARE_SIZE = WARP_SIZE // GRID_SIZE

def draw_grid_lines(frame):
    for i in range(1, GRID_SIZE):
        # Vertical
        cv2.line(frame, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, WARP_SIZE), (255, 0, 0), 1)
        # Horizontal
        cv2.line(frame, (0, i * SQUARE_SIZE), (WARP_SIZE, i * SQUARE_SIZE), (255, 0, 0), 1)

def draw_grid_labels(frame):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            label = f"{chr(ord('a') + col)}{8 - row}"
            x = col * SQUARE_SIZE + 5
            y = row * SQUARE_SIZE + 20
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
