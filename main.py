import os
import cv2
import numpy as np
import time

from utils.delta_recorder import DeltaRecorder
from utils.data_saver import DataSaver
from eye_tracker.gaze_tracking import GazeTracking
from utils.head_pose_utils import get_head_pose
from utils.distance_utils import estimate_distance_cm

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
GRID_X = 16
GRID_Y = 8
N_GRID = GRID_X * GRID_Y
BORDER_RATIO = 0.08

SPLIT_SEQUENCE = [
    ("y", 1),  # Y1
    ("x", 1),  # X1
    ("y", 2),  # Y2
    ("x", 2),  # X2
    ("y", 3),  # Y3
    ("x", 3),  # X3
    ("y", 4),  # Y4
]

def get_hierarchical_labels(gx, gy, split_sequence, screen_width, screen_height):
    x_bounds = (0, screen_width)
    y_bounds = (0, screen_height)
    x_regions = [x_bounds]
    y_regions = [y_bounds]
    labels = []
    for axis, _ in split_sequence:
        if axis == "y":
            new_regions = []
            label = None
            for (x0, x1) in x_regions:
                mid = (x0 + x1) / 2
                if x0 <= gx < mid:
                    new_regions.append((x0, mid))
                    if label is None:
                        label = 1
                else:
                    new_regions.append((mid, x1))
                    if label is None:
                        label = 0
            x_regions = new_regions
            labels.append(label)
        elif axis == "x":
            new_regions = []
            label = None
            for (y0, y1) in y_regions:
                mid = (y0 + y1) / 2
                if y0 <= gy < mid:
                    new_regions.append((y0, mid))
                    if label is None:
                        label = 1
                else:
                    new_regions.append((mid, y1))
                    if label is None:
                        label = 0
            y_regions = new_regions
            labels.append(label)
    return labels

def generate_grid_points():
    points = []
    cell_w = SCREEN_WIDTH / GRID_X
    cell_h = SCREEN_HEIGHT / GRID_Y
    for row in range(GRID_Y):
        for col in range(GRID_X):
            x0 = int(col * cell_w + BORDER_RATIO * cell_w)
            x1 = int((col + 1) * cell_w - BORDER_RATIO * cell_w)
            y0 = int(row * cell_h + BORDER_RATIO * cell_h)
            y1 = int((row + 1) * cell_h - BORDER_RATIO * cell_h)
            x = np.random.randint(x0, x1 + 1)
            y = np.random.randint(y0, y1 + 1)
            grid_id = row * GRID_X + col
            points.append((x, y, grid_id, col, row))
    np.random.shuffle(points)
    return points

def draw_grid(img):
    cell_w = SCREEN_WIDTH / GRID_X
    cell_h = SCREEN_HEIGHT / GRID_Y
    for i in range(GRID_X + 1):
        x = int(i * cell_w)
        cv2.line(img, (x, 0), (x, SCREEN_HEIGHT), (60, 60, 60), 1)
    for j in range(GRID_Y + 1):
        y = int(j * cell_h)
        cv2.line(img, (0, y), (SCREEN_WIDTH, y), (60, 60, 60), 1)

def ask_continue_screen():
    img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
    text = "Yeni bloğa devam etmek istiyor musunuz?"
    text2 = "SPACE: Devam   ESC: Çıkış"
    cv2.putText(img, text, (180, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 6)
    cv2.putText(img, text2, (320, 580), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
    cv2.namedWindow("GazeData", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("GazeData", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("GazeData", img)
    while True:
        key = cv2.waitKey(0)
        if key == 32:  # SPACE
            cv2.destroyWindow("GazeData")
            return True
        elif key == 27:  # ESC
            cv2.destroyWindow("GazeData")
            return False

def main():
    save_path = "data/gaze_train_128.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    recorder = DataSaver(save_path)
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(1)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    
    while True:
        # --- BLOK BAŞI: Pattern ID ve DeltaRecorder resetlenir ---
        pattern_id = int(time.time())
        delta_recorder = DeltaRecorder()  # Yeniden başlat!
        all_points = generate_grid_points()

        win_name = "GazeData"
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for i, (x, y, grid_id, col, row) in enumerate(all_points):
            img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
            draw_grid(img)
            cv2.circle(img, (x, y), 30, (0, 255, 0), -1)
            cv2.putText(img, f"{i + 1}/{len(all_points)}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (220, 220, 255), 4)

            cv2.imshow(win_name, img)
            while True:
                key = cv2.waitKey(1)
                if key == 13 or key == 10:  # ENTER
                    ret, frame = webcam.read()
                    if not ret:
                        print("Kamera hatası!")
                        continue
                    gaze.refresh(frame)
                    if gaze.is_blinking():
                        print("Gözler kapalı, tekrar deneyin (blink algılandı).")
                        continue
                    left_pupil = gaze.pupil_left_coords()
                    right_pupil = gaze.pupil_right_coords()
                    if left_pupil is None or right_pupil is None:
                        print("Göz/pupil algılanamadı! Lütfen tekrar deneyin.")
                        continue
                    iris_x = (left_pupil[0] + right_pupil[0]) / 2
                    iris_y = (left_pupil[1] + right_pupil[1]) / 2
                    pitch, yaw = get_head_pose(frame)
                    if pitch is None or yaw is None:
                        print("Baş pozisyonu bulunamadı, tekrar deneyin!")
                        continue
                    distance = estimate_distance_cm(frame)
                    if distance is None:
                        print("Mesafe ölçülemedi, tekrar deneyin!")
                        continue

                    record = delta_recorder.record(
                        current_iris=(iris_x, iris_y),
                        distance_cm=distance,
                        angle_id="0",
                        pattern_id=str(pattern_id),
                        head_pitch=pitch,
                        head_yaw=yaw,
                        grid_id=grid_id,
                        screen_width=SCREEN_WIDTH,
                        screen_height=SCREEN_HEIGHT
                    )
                    hier_labels = get_hierarchical_labels(
                        x, y, SPLIT_SEQUENCE, SCREEN_WIDTH, SCREEN_HEIGHT
                    )
                    try:
                        record["v_ratio"] = gaze.vertical_ratio()
                    except Exception:
                        record["v_ratio"] = None
                    try:
                        record["h_ratio"] = gaze.horizontal_ratio()
                    except Exception:
                        record["h_ratio"] = None

                    record["y1"] = hier_labels[0]
                    record["x1"] = hier_labels[1]
                    record["y2"] = hier_labels[2]
                    record["x2"] = hier_labels[3]
                    record["y3"] = hier_labels[4]
                    record["x3"] = hier_labels[5]
                    record["y4"] = hier_labels[6]

                    recorder.save_single(record)
                    break
                elif key == 27:  # ESC
                    cv2.destroyAllWindows()
                    recorder.close()
                    webcam.release()
                    return

        cv2.destroyAllWindows()
        if not ask_continue_screen():
            recorder.close()
            webcam.release()
            break

if __name__ == "__main__":
    main()
