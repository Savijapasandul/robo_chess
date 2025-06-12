import cv2
import numpy as np

WARP_SIZE = 400  # Output resolution of warped board

def extract_corners_from_box(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # TL, TR, BR, BL

def draw_corners(frame, corners):
    for i, pt in enumerate(corners):
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"P{i+1}", (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def draw_center(frame, center):
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    cv2.putText(frame, "Center", (center[0]+5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def get_homography(corners):
    src = np.array(corners, dtype=np.float32)
    dst = np.array([
        [0, 0],              # A1 (top-left)
        [WARP_SIZE, 0],      # H1
        [WARP_SIZE, WARP_SIZE],  # H8
        [0, WARP_SIZE]       # A8
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H

def warp_board(frame, H):
    return cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
