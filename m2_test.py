import sys
import pickle
import numpy as np
import cv2
import math

from collections import Counter

from eye_tracker.gaze_tracking import GazeTracking
from utils.head_pose_utils import get_head_pose
from utils.distance_utils import estimate_distance_cm
from PyQt5 import QtWidgets, QtGui, QtCore

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
GRID_X, GRID_Y = 2, 2  # 2x2 grid

def grid_center(grid_id, grid_x, grid_y):
    cell_w = SCREEN_WIDTH / grid_x
    cell_h = SCREEN_HEIGHT / grid_y
    gx = grid_id % grid_x
    gy = grid_id // grid_x
    x = int(gx * cell_w + cell_w / 2)
    y = int(gy * cell_h + cell_h / 2)
    return x, y

def mode_or_first(values):
    counts = Counter(values)
    most_common = counts.most_common()
    return most_common[0][0]

class GazeOverlay(QtWidgets.QWidget):
    def __init__(self, model1, scaler1, model2, scaler2):
        super().__init__()
        self.model1 = model1
        self.scaler1 = scaler1
        self.model2 = model2
        self.scaler2 = scaler2

        self.tracker = GazeTracking()
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

        self.prev_iris = None
        self.prev_head_pitch = None
        self.prev_head_yaw = None

        self.est_x = SCREEN_WIDTH // 2
        self.est_y = SCREEN_HEIGHT // 2

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
        self.timer.start(500)

    def update_gaze(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı!")
                return

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

                screen_center_x = SCREEN_WIDTH / 2
                screen_center_y = SCREEN_HEIGHT / 2

                h_ratio = self.tracker.horizontal_ratio() if self.tracker.horizontal_ratio() is not None else 0.0
                v_ratio = self.tracker.vertical_ratio() if self.tracker.vertical_ratio() is not None else 0.0

                features_m1 = [
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

                # Model1: Majority voting (Y1, sol/sağ)
                features_m1_scaled = self.scaler1.transform([features_m1])
                y1_preds = [int(self.model1.predict(features_m1_scaled)[0]) for _ in range(3)]
                y1_majority = mode_or_first(y1_preds)

                # Model2: Majority voting (X1, üst/alt)
                features_m2_scaled = self.scaler2.transform([features_m1])[0]
                features_m2 = np.concatenate([features_m2_scaled, [y1_majority]])
                x1_preds = [int(self.model2.predict([features_m2])[0]) for _ in range(3)]
                x1_majority = mode_or_first(x1_preds)

                # DOĞRU GRID MAPPING
                # y1_majority: 1=sol, 0=sağ
                # x1_majority: 1=üst, 0=alt
                row = 1 - x1_majority   # x1=1 ise üst (row=0), x1=0 ise alt (row=1)
                col = 1 - y1_majority   # y1=1 ise sol (col=0), y1=0 ise sağ (col=1)
                grid_id = row * 2 + col

                x_c, y_c = grid_center(grid_id, GRID_X, GRID_Y)
                self.est_x = x_c
                self.est_y = y_c

                self.prev_iris = (iris_x, iris_y)
                self.prev_head_pitch = pitch
                self.prev_head_yaw = yaw

            self.update()

        except Exception as e:
            print(f"[GazeOverlay Hata]: {e}")

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.green, 4))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.green))
        painter.drawEllipse(QtCore.QPointF(self.est_x, self.est_y), 30, 30)

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

    overlay = GazeOverlay(model1, scaler1, model2, scaler2)
    overlay.showFullScreen()
    sys.exit(app.exec_())
