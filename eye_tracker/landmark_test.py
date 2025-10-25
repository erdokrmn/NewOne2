import cv2
import dlib
import numpy as np
import sys
import os
import json
import time
from collections import deque
from eye_tracker.gaze_tracking import GazeTracking
from utils.distance_utils import estimate_distance_cm

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
CIRCLE_R = 40

# 4 köşe ve orta nokta
test_points = [
    (80, 80),  # sol üst
    (SCREEN_WIDTH-80, 80),  # sağ üst
    (80, SCREEN_HEIGHT-80),  # sol alt
    (SCREEN_WIDTH-80, SCREEN_HEIGHT-80),  # sağ alt
    (SCREEN_WIDTH//2, SCREEN_HEIGHT//2),  # orta
]

output_jsonl = "h_ratio_true_pupil_results.jsonl"

gaze = GazeTracking()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("eye_tracker/trained_models/shape_predictor_68_face_landmarks.dat")

# Smoothing için kuyruk
smooth_n = 10
h_ratio_q, v_ratio_q = deque(maxlen=smooth_n), deque(maxlen=smooth_n)

def find_pupil_in_eye(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    (x, y), _ = cv2.minEnclosingCircle(max_contour)
    return int(x), int(y)

def normalized_pupil_coords(pupil_xy, bbox):
    if pupil_xy is None:
        return 0.5, 0.5
    x0, y0, x1, y1 = bbox
    w, h = x1-x0, y1-y0
    rel_x = (pupil_xy[0]) / w if w != 0 else 0.5
    rel_y = (pupil_xy[1]) / h if h != 0 else 0.5
    return rel_x, rel_y

cap = cv2.VideoCapture(0)
cv2.namedWindow("h_ratio_game", cv2.WINDOW_NORMAL)
current_point = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Nokta çiz
    x, y = test_points[current_point]
    cv2.circle(frame, (x, y), CIRCLE_R, (0,255,255), -1)
    cv2.putText(frame, f"Point {current_point+1}/{len(test_points)}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,220,0), 3)
    cv2.putText(frame, "ENTER: Kayit & Sonraki, ESC: Bitir", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (80,80,255), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    info_text = ""
    status_ok = False
    distance_cm = None

    for face in faces:
        shape = predictor(gray, face)

        # Landmarkları çiz
        for i in range(68):
            lx, ly = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (lx, ly), 1, (255, 200, 50), -1)

        # Sol/sağ göz kutusu
        left_eye_pts = [(shape.part(n).x, shape.part(n).y) for n in range(36, 42)]
        right_eye_pts = [(shape.part(n).x, shape.part(n).y) for n in range(42, 48)]
        lx0 = min(x for x, y in left_eye_pts)
        ly0 = min(y for x, y in left_eye_pts)
        lx1 = max(x for x, y in left_eye_pts)
        ly1 = max(y for x, y in left_eye_pts)
        rx0 = min(x for x, y in right_eye_pts)
        ry0 = min(y for x, y in right_eye_pts)
        rx1 = max(x for x, y in right_eye_pts)
        ry1 = max(y for x, y in right_eye_pts)

        left_eye_img = frame[ly0:ly1, lx0:lx1]
        right_eye_img = frame[ry0:ry1, rx0:rx1]
        left_pupil = find_pupil_in_eye(left_eye_img) if left_eye_img.size > 0 else None
        right_pupil = find_pupil_in_eye(right_eye_img) if right_eye_img.size > 0 else None

        # Normalize et (0-1 arası)
        h_left, v_left = normalized_pupil_coords(left_pupil, (lx0, ly0, lx1, ly1))
        h_right, v_right = normalized_pupil_coords(right_pupil, (rx0, ry0, rx1, ry1))
        h_ratio = (h_left + h_right) / 2
        v_ratio = (v_left + v_right) / 2

        # Smoothing
        h_ratio_q.append(h_ratio)
        v_ratio_q.append(v_ratio)
        h_ratio_smooth = np.mean(h_ratio_q)
        v_ratio_smooth = np.mean(v_ratio_q)

        # Distance
        distance_cm = estimate_distance_cm(frame)
        info_text = f"h={h_ratio_smooth:.3f}, v={v_ratio_smooth:.3f}, d={distance_cm:.1f}cm"
        status_ok = True

        # Çizimler
        cv2.rectangle(frame, (lx0, ly0), (lx1, ly1), (0,255,0), 1)
        cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0,255,0), 1)
        if left_pupil:
            cv2.circle(frame, (lx0 + left_pupil[0], ly0 + left_pupil[1]), 6, (0,0,255), -1)
        if right_pupil:
            cv2.circle(frame, (rx0 + right_pupil[0], ry0 + right_pupil[1]), 6, (0,0,255), -1)
        break

    color = (0,220,0) if status_ok else (0,0,255)
    cv2.putText(frame, info_text, (30,180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
    cv2.imshow("h_ratio_game", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 13 or key == 10:
        if status_ok:
            record = {
                "test_point": current_point,
                "screen_x": x,
                "screen_y": y,
                "timestamp": int(time.time()),
                "h_ratio": float(h_ratio_smooth),
                "v_ratio": float(v_ratio_smooth),
                "left_pupil": left_pupil,
                "right_pupil": right_pupil,
                "distance_cm": float(distance_cm) if distance_cm else None
            }
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Kayıt edildi: {record}")
        current_point += 1
        if current_point >= len(test_points):
            print("Tüm noktalar test edildi. Çıkılıyor.")
            break

cap.release()
cv2.destroyAllWindows()
