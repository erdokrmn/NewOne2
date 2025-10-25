import sys
import pickle
import numpy as np
import cv2
import math
import time
from collections import Counter

from eye_tracker.gaze_tracking import GazeTracking
from utils.head_pose_utils import get_head_pose
from utils.distance_utils import estimate_distance_cm
from PyQt5 import QtWidgets, QtGui, QtCore

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

def mode_or_first(values):
    counts = Counter(values)
    most_common = counts.most_common()
    return most_common[0][0]

def grid_center(y1, y2, x1, x2, y3, x3):
    half_w = SCREEN_WIDTH // 2
    if y1 == 1:
        x_left = 0
        x_right = half_w
    else:
        x_left = half_w
        x_right = SCREEN_WIDTH

    quarter_w = (x_right - x_left) // 2
    if y2 == 1:
        x0 = x_left
        x1_ = x_left + quarter_w
    else:
        x0 = x_left + quarter_w
        x1_ = x_right

    half_h = SCREEN_HEIGHT // 2
    if x1 == 1:
        y_top = 0
        y_bot = half_h
    else:
        y_top = half_h
        y_bot = SCREEN_HEIGHT

    quarter_h = (y_bot - y_top) // 2
    if x2 == 1:
        y0 = y_top
        y1__ = y_top + quarter_h
    else:
        y0 = y_top + quarter_h
        y1__ = y_bot

    small_w = (x1_ - x0) // 2
    if y3 == 1:
        fx0 = x0
        fx1 = x0 + small_w
    else:
        fx0 = x0 + small_w
        fx1 = x1_

    small_h = (y1__ - y0) // 2
    if x3 == 1:
        fy0 = y0
        fy1 = y0 + small_h
    else:
        fy0 = y0 + small_h
        fy1 = y1__

    cx = (fx0 + fx1) // 2
    cy = (fy0 + fy1) // 2
    return cx, cy

class GazeOverlay(QtWidgets.QWidget):
    def __init__(self, model1, scaler1, model2, scaler2, model3, scaler3, model4, scaler4, model5, scaler5, model6, scaler6):
        super().__init__()
        self.model1 = model1
        self.scaler1 = scaler1
        self.model2 = model2
        self.scaler2 = scaler2
        self.model3 = model3
        self.scaler3 = scaler3
        self.model4 = model4
        self.scaler4 = scaler4
        self.model5 = model5
        self.scaler5 = scaler5
        self.model6 = model6
        self.scaler6 = scaler6

        self.tracker = GazeTracking()
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

        self.prev_iris = None
        self.prev_head_pitch = None
        self.prev_head_yaw = None

        self.est_x = None
        self.est_y = None

        self.setGeometry(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_gaze)
        self.timer.start(100)

    def collect_nframes(self, n=3, max_time_ms=1000):
        frames = []
        start_time = time.time()
        while len(frames) < n:
            if (time.time() - start_time) * 1000 > max_time_ms:
                break
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.tracker.refresh(frame)
            left_pupil = self.tracker.pupil_left_coords()
            right_pupil = self.tracker.pupil_right_coords()
            if left_pupil is not None and right_pupil is not None:
                iris_x = (left_pupil[0] + right_pupil[0]) / 2
                iris_y = (left_pupil[1] + right_pupil[1]) / 2
                pitch, yaw = get_head_pose(frame)
                distance = estimate_distance_cm(frame)
                delta_iris_x = iris_x - self.prev_iris[0] if self.prev_iris else 0.0
                delta_iris_y = iris_y - self.prev_iris[1] if self.prev_iris else 0.0
                delta_head_pitch = (pitch - self.prev_head_pitch) if self.prev_head_pitch is not None and pitch is not None else 0.0
                delta_head_yaw = (yaw - self.prev_head_yaw) if self.prev_head_yaw is not None and yaw is not None else 0.0

                # DELTA FİLTRELERİ
                if not (-40 <= delta_iris_x <= 40):
                    continue
                if not (-20 <= delta_iris_y <= 20):
                    continue
                if not (-10 <= delta_head_pitch <= 10):
                    continue
                if not (-30 <= delta_head_yaw <= 30):
                    continue

                screen_center_x = SCREEN_WIDTH / 2
                screen_center_y = SCREEN_HEIGHT / 2
                h_ratio = self.tracker.horizontal_ratio() if self.tracker.horizontal_ratio() is not None else 0.0
                v_ratio = self.tracker.vertical_ratio() if self.tracker.vertical_ratio() is not None else 0.0
                features = [
                    iris_x, iris_y, distance or 0.0,
                    pitch or 0.0, yaw or 0.0,
                    delta_iris_x, delta_iris_y,
                    delta_head_pitch, delta_head_yaw,
                    (distance or 0.0) / SCREEN_WIDTH,
                    iris_x / SCREEN_WIDTH, iris_y / SCREEN_HEIGHT,
                    (iris_x - screen_center_x) / SCREEN_WIDTH,
                    (iris_y - screen_center_y) / SCREEN_HEIGHT,
                    ((iris_x - screen_center_x) ** 2 + (iris_y - screen_center_y) ** 2) ** 0.5,
                    math.atan2(iris_y - screen_center_y, iris_x - screen_center_x),
                    h_ratio,
                    v_ratio,
                ]
                frames.append(features)
                self.prev_iris = (iris_x, iris_y)
                self.prev_head_pitch = pitch
                self.prev_head_yaw = yaw
        return frames

    def update_gaze(self):
        try:
            all_features = self.collect_nframes(n=3, max_time_ms=1000)
            if len(all_features) < 3:
                self.est_x = None
                self.est_y = None
                self.update()
                return

            # Model1 (Y1)
            y1_votes = []
            for features in all_features:
                features_scaled = self.scaler1.transform([features])
                y1_pred = int(self.model1.predict(features_scaled)[0])
                y1_votes.append(y1_pred)
            y1_majority = mode_or_first(y1_votes)

            # Model2 (X1)
            x1_votes = []
            for features in all_features:
                features_scaled = self.scaler2.transform([features])[0]
                features_m2 = np.concatenate([features_scaled, [y1_majority]])
                x1_pred = int(self.model2.predict([features_m2])[0])
                x1_votes.append(x1_pred)
            x1_majority = mode_or_first(x1_votes)

            # Model3 (Y2)
            y2_votes = []
            for features in all_features:
                features_scaled = self.scaler3.transform([features])[0]
                features_m3 = np.concatenate([features_scaled, [y1_majority, x1_majority]])
                y2_pred = int(self.model3.predict([features_m3])[0])
                y2_votes.append(y2_pred)
            y2_majority = mode_or_first(y2_votes)

            # Model4 (X2)
            x2_votes = []
            for features in all_features:
                features_scaled = self.scaler4.transform([features])[0]
                features_m4 = np.concatenate([features_scaled, [y1_majority, x1_majority, y2_majority]])
                x2_pred = int(self.model4.predict([features_m4])[0])
                x2_votes.append(x2_pred)
            x2_majority = mode_or_first(x2_votes)

            # Model5 (Y3)
            y3_votes = []
            for features in all_features:
                features_scaled = self.scaler5.transform([features])[0]
                features_m5 = np.concatenate([features_scaled, [y1_majority, x1_majority, y2_majority, x2_majority]])
                y3_pred = int(self.model5.predict([features_m5])[0])
                y3_votes.append(y3_pred)
            y3_majority = mode_or_first(y3_votes)

            # Model6 (X3)
            x3_votes = []
            for features in all_features:
                features_scaled = self.scaler6.transform([features])[0]
                features_m6 = np.concatenate([features_scaled, [y1_majority, x1_majority, y2_majority, x2_majority, y3_majority]])
                x3_pred = int(self.model6.predict([features_m6])[0])
                x3_votes.append(x3_pred)
            x3_majority = mode_or_first(x3_votes)

            x_c, y_c = grid_center(
                y1_majority, y2_majority, x1_majority, x2_majority, y3_majority, x3_majority
            )
            self.est_x = x_c
            self.est_y = y_c

            self.update()

        except Exception as e:
            print(f"[GazeOverlay Hata]: {e}")

    def paintEvent(self, event):
        if self.est_x is None or self.est_y is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.green, 4))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.green))
        painter.drawEllipse(QtCore.QPointF(self.est_x, self.est_y), 20, 20)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            print("👋 ESC ile çıkılıyor.")
            self.cap.release()
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    with open("data/gaze_grid_m1.pkl", "rb") as f:
        model1_data = pickle.load(f)
    model1 = model1_data["model"]
    scaler1 = model1_data["scaler"]

    with open("data/gaze_grid_m2.pkl", "rb") as f:
        model2_data = pickle.load(f)
    model2 = model2_data["model"]
    scaler2 = model2_data["scaler"]

    with open("data/gaze_grid_m3.pkl", "rb") as f:
        model3_data = pickle.load(f)
    model3 = model3_data["model"]
    scaler3 = model3_data["scaler"]

    with open("data/gaze_grid_m4.pkl", "rb") as f:
        model4_data = pickle.load(f)
    model4 = model4_data["model"]
    scaler4 = model4_data["scaler"]

    with open("data/gaze_grid_m5.pkl", "rb") as f:
        model5_data = pickle.load(f)
    model5 = model5_data["model"]
    scaler5 = model5_data["scaler"]

    with open("data/gaze_grid_m6.pkl", "rb") as f:
        model6_data = pickle.load(f)
    model6 = model6_data["model"]
    scaler6 = model6_data["scaler"]

    overlay = GazeOverlay(
        model1, scaler1,
        model2, scaler2,
        model3, scaler3,
        model4, scaler4,
        model5, scaler5,
        model6, scaler6
    )
    overlay.showFullScreen()
    sys.exit(app.exec_())
