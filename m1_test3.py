import sys
import pickle
import numpy as np
import cv2
import math

from eye_tracker.gaze_tracking import GazeTracking
from utils.head_pose_utils import get_head_pose
from utils.distance_utils import estimate_distance_cm
from PyQt5 import QtWidgets, QtGui, QtCore

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
screen_center_x = SCREEN_WIDTH / 2
screen_center_y = SCREEN_HEIGHT / 2
screen_diag = ((SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2) ** 0.5)
reference_points = [
    (0, 0),                             # sol üst
    (SCREEN_WIDTH, 0),                  # sağ üst
    (0, SCREEN_HEIGHT),                 # sol alt
    (SCREEN_WIDTH, SCREEN_HEIGHT),      # sağ alt
]

def grid_center_by_y1(y1_value):
    cell_w = SCREEN_WIDTH // 2
    y = SCREEN_HEIGHT // 2
    if y1_value == 1:
        x = cell_w // 2
    else:
        x = cell_w + cell_w // 2
    return x, y

class GazeOverlay(QtWidgets.QWidget):
    def __init__(self, model, scaler):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.tracker = GazeTracking()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
        self.prev_iris = None
        self.last_pitch = None
        self.last_yaw = None
        self.est_x = SCREEN_WIDTH // 2
        self.est_y = SCREEN_HEIGHT // 2
        self.last_est_x = self.est_x
        self.last_est_y = self.est_y

        # Eşikler (değiştirebilirsin)
        self.DELTA_THRESH_IRIS_X = 3
        self.DELTA_THRESH_IRIS_Y = 3
        self.DELTA_THRESH_PITCH = 0.4
        self.DELTA_THRESH_YAW = 0.6

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
        self.timer.start(300)

    def update_gaze(self):
        try:
            ret, frame_cam = self.cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı!")
                return

            self.tracker.refresh(frame_cam)
            left_pupil = self.tracker.pupil_left_coords()
            right_pupil = self.tracker.pupil_right_coords()

            if left_pupil is not None and right_pupil is not None:
                iris_x = (left_pupil[0] + right_pupil[0]) / 2
                iris_y = (left_pupil[1] + right_pupil[1]) / 2
                pitch, yaw = get_head_pose(frame_cam)
                distance = estimate_distance_cm(frame_cam)
                h_ratio = self.tracker.horizontal_ratio() if self.tracker.horizontal_ratio() is not None else 0.0
                v_ratio = self.tracker.vertical_ratio() if self.tracker.vertical_ratio() is not None else 0.0

                if self.prev_iris is not None:
                    delta_iris_x = iris_x - self.prev_iris[0]
                    delta_iris_y = iris_y - self.prev_iris[1]
                else:
                    delta_iris_x = 0.0
                    delta_iris_y = 0.0

                if self.last_pitch is not None and pitch is not None:
                    delta_head_pitch = pitch - self.last_pitch
                else:
                    delta_head_pitch = 0.0
                if self.last_yaw is not None and yaw is not None:
                    delta_head_yaw = yaw - self.last_yaw
                else:
                    delta_head_yaw = 0.0

                # 4 köşe oranı
                corner_ratios = [
                    ((iris_x - ref[0]) ** 2 + (iris_y - ref[1]) ** 2) ** 0.5 / screen_diag
                    for ref in reference_points
                ]

                # Eğer hareket düşükse tahmin güncelleme, son konumu koru
                if (abs(delta_iris_x) < self.DELTA_THRESH_IRIS_X and
                    abs(delta_iris_y) < self.DELTA_THRESH_IRIS_Y and
                    abs(delta_head_pitch) < self.DELTA_THRESH_PITCH and
                    abs(delta_head_yaw) < self.DELTA_THRESH_YAW):
                    self.est_x = self.last_est_x
                    self.est_y = self.last_est_y
                    self.prev_iris = (iris_x, iris_y)
                    self.last_pitch = pitch
                    self.last_yaw = yaw
                    self.update()
                    return

                features = [
                    iris_x,
                    iris_y,
                    distance if distance is not None else 0.0,
                    pitch if pitch is not None else 0.0,
                    yaw if yaw is not None else 0.0,
                    delta_iris_x,
                    delta_iris_y,
                    delta_head_pitch,
                    delta_head_yaw,
                    ((iris_x - screen_center_x) ** 2 + (iris_y - screen_center_y) ** 2) ** 0.5,
                    math.atan2(iris_y - screen_center_y, iris_x - screen_center_x),
                    h_ratio,
                    v_ratio,
                    *corner_ratios
                ]
                features_scaled = self.scaler.transform([features])
                y1_pred = int(self.model.predict(features_scaled)[0])
                x_c, y_c = grid_center_by_y1(y1_pred)
                self.est_x = x_c
                self.est_y = y_c
                self.last_est_x = self.est_x
                self.last_est_y = self.est_y

                self.prev_iris = (iris_x, iris_y)
                self.last_pitch = pitch
                self.last_yaw = yaw

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
        model_data = pickle.load(f)
    model = model_data["model"]
    scaler = model_data["scaler"]
    overlay = GazeOverlay(model, scaler)
    overlay.showFullScreen()
    sys.exit(app.exec_())
