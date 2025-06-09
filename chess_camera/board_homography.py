# === board_homography.py ===
import numpy as np
import cv2

def get_warped_board(frame, src_pts):
    dst_pts = np.float32([[0,0], [800,0], [800,800], [0,800]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, matrix, (800, 800))
    return warped