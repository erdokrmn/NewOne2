import sys
import pickle
import numpy as np
import cv2
import math

from eye_tracker.gaze_tracking import GazeTracking
from utils.head_pose_utils import get_head_pose
from utils.distance_utils import estimate_distance_cm
from PyQt5 import QtWidgets, QtGui, QtCore

# Grid ve ekran ayarları
GRID_X, GRID_Y = 2, 1  # 2 grid için
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

def grid_center_by_y1(y1_value):
    # y1: 1 = sol, 0 = sağ
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
        self.cap = cv2.VideoCapture(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

        self.prev_iris = None
        self.last_iris = None
        self.last_pitch = None
        self.last_yaw = None

        self.est_x = SCREEN_WIDTH // 2
        self.est_y = SCREEN_HEIGHT // 2

        self.iris_move_thresh = 12    # px
        self.head_move_thresh = 20     # derece

        self.iris_history = []
        self.smooth_n = 1

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
        self.timer.start(500)  # ms

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

                # Ratios
                h_ratio = self.tracker.horizontal_ratio() if self.tracker.horizontal_ratio() is not None else 0.0
                v_ratio = self.tracker.vertical_ratio() if self.tracker.vertical_ratio() is not None else 0.0

                # Delta iris
                if self.prev_iris is not None:
                    delta_iris_x = iris_x - self.prev_iris[0]
                    delta_iris_y = iris_y - self.prev_iris[1]
                else:
                    delta_iris_x = 0.0
                    delta_iris_y = 0.0

                # Delta Head Pitch/Yaw
                if self.last_pitch is not None and pitch is not None:
                    delta_head_pitch = pitch - self.last_pitch
                else:
                    delta_head_pitch = 0.0
                if self.last_yaw is not None and yaw is not None:
                    delta_head_yaw = yaw - self.last_yaw
                else:
                    delta_head_yaw = 0.0

                # Normalizasyonlar
                distance_norm = distance / SCREEN_WIDTH if distance is not None else 0.0
                iris_x_norm = iris_x / SCREEN_WIDTH
                iris_y_norm = iris_y / SCREEN_HEIGHT

                screen_center_x = SCREEN_WIDTH / 2
                screen_center_y = SCREEN_HEIGHT / 2
                dx_norm = (iris_x - screen_center_x) / SCREEN_WIDTH
                dy_norm = (iris_y - screen_center_y) / SCREEN_HEIGHT
                distance_to_center = ((iris_x - screen_center_x) ** 2 + (iris_y - screen_center_y) ** 2) ** 0.5
                angle_to_center = math.atan2(iris_y - screen_center_y, iris_x - screen_center_x)

                # --- TAM SIRALI FEATURE VEKTÖRÜ ---
                features = [
                    iris_x,             # 1
                    iris_y,             # 2
                    distance if distance is not None else 0.0,     # 3
                    pitch if pitch is not None else 0.0,           # 4
                    yaw if yaw is not None else 0.0,               # 5
                   # delta_iris_x,       # 6
                   # delta_iris_y,       # 7
                   # delta_head_pitch,   # 8
                   # delta_head_yaw,     # 9
                    distance_norm,      # 10
                    iris_x_norm,        # 11
                    iris_y_norm,        # 12
                    dx_norm,            # 13
                    dy_norm,            # 14
                    distance_to_center, # 15
                    angle_to_center,    # 16
                    h_ratio,            # 17
                    v_ratio,            # 18
                ]
                # ---------------------------------

                features_scaled = self.scaler.transform([features])
                y1_pred = int(self.model.predict(features_scaled)[0])  # Model çıktısı y1 etiketi (1=sol, 0=sağ)
                x_c, y_c = grid_center_by_y1(y1_pred)
                self.est_x = x_c
                self.est_y = y_c

                self.prev_iris = (iris_x, iris_y)

            self.last_iris = (iris_x, iris_y)
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
