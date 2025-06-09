# === square_mapping.py ===
def get_square(x, y, board_size=800):
    col = int(x // (board_size / 8))
    row = int(y // (board_size / 8))
    file = chr(ord('a') + col)
    rank = 8 - row
    return f"{file}{rank}"

def map_detections_to_squares(detections):
    return [(piece, get_square(x, y)) for piece, x, y in detections]