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
    (0, 0),
    (SCREEN_WIDTH, 0),
    (0, SCREEN_HEIGHT),
    (SCREEN_WIDTH, SCREEN_HEIGHT),
]

def grid_center(y1, x1):
    x_half = SCREEN_WIDTH // 2
    y_half = SCREEN_HEIGHT // 2
    x = x_half // 2 if y1 == 1 else x_half + x_half // 2
    y = y_half // 2 if x1 == 1 else y_half + y_half // 2
    return x, y

class GazeOverlay(QtWidgets.QWidget):
    def __init__(self, model_m2, scaler_m2, model_m1, scaler_m1):
        super().__init__()
        self.model_m2 = model_m2
        self.scaler_m2 = scaler_m2
        self.model_m1 = model_m1
        self.scaler_m1 = scaler_m1
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

        # Eşikler
        self.DELTA_THRESH_IRIS_X = 3
        self.DELTA_THRESH_IRIS_Y = 3
        self.DELTA_THRESH_PITCH = 0.4
        self.DELTA_THRESH_YAW = 0.6

        self.setGeometry(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.Tool)
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
                h_ratio = self.tracker.horizontal_ratio() or 0.0
                v_ratio = self.tracker.vertical_ratio() or 0.0

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

                # HAREKET AZSA SON TAHMİNİ KULLAN
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
                    iris_x, iris_y, distance or 0.0, pitch or 0.0, yaw or 0.0,
                    delta_iris_x, delta_iris_y, delta_head_pitch, delta_head_yaw,
                    ((iris_x - screen_center_x) ** 2 + (iris_y - screen_center_y) ** 2) ** 0.5,
                    math.atan2(iris_y - screen_center_y, iris_x - screen_center_x),
                    h_ratio, v_ratio, *corner_ratios
                ]

                # Model 1 ile y1 tahmini
                x_input_m1 = self.scaler_m1.transform([features])
                y1 = int(self.model_m1.predict(x_input_m1)[0])

                # Model 2 ile x1 tahmini
                features_with_y1 = features + [y1]
                x_input_m2 = self.scaler_m2.transform([features_with_y1])
                x1 = int(self.model_m2.predict(x_input_m2)[0])

                x, y = grid_center(y1, x1)
                self.est_x = x
                self.est_y = y
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
        m1_data = pickle.load(f)
    model_m1 = m1_data["model"]
    scaler_m1 = m1_data["scaler"]

    with open("data/gaze_grid_m2.pkl", "rb") as f:
        m2_data = pickle.load(f)
    model_m2 = m2_data["model"]
    scaler_m2 = m2_data["scaler"]

    overlay = GazeOverlay(model_m2, scaler_m2, model_m1, scaler_m1)
    overlay.showFullScreen()
    sys.exit(app.exec_())
