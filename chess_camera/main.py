from camera_input import get_camera_frame
from board_homography import get_warped_board
from piece_detection import detect_pieces
from square_mapping import map_detections_to_squares
import numpy as np
import flet as ft
import threading
import time

SRC_POINTS = np.float32([[100, 100], [700, 100], [700, 700], [100, 700]])

def run_detection_loop(pub):
    while True:
        try:
            frame = get_camera_frame()
            warped = get_warped_board(frame, SRC_POINTS)
            detections = detect_pieces(warped)
            pieces = map_detections_to_squares(detections)
            pub.send_all(pieces)
            time.sleep(1)  # Small delay to avoid overloading
        except Exception as e:
            print("Detection error:", e)

def main(page: ft.Page):
    # PubSub setup
    pub = ft.PubSubHub()
    page.pubsub = pub

    # Start detection in a background thread
    threading.Thread(target=run_detection_loop, args=(pub,), daemon=True).start()

    # Import and render the board UI
    from gui_display import main as board_main
    board_main(page)

ft.app(target=main)
