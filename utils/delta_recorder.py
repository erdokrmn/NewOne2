from typing import Tuple, Dict, Optional
import math

class DeltaRecorder:
    def __init__(self):
        self.prev_iris: Optional[Tuple[float, float]] = None
        self.prev_pitch: Optional[float] = None
        self.prev_yaw: Optional[float] = None

    def reset(self):
        self.prev_iris = None
        self.prev_pitch = None
        self.prev_yaw = None

    def record(
        self,
        current_iris: Tuple[float, float],
        distance_cm: float,
        angle_id: str,
        pattern_id: str,
        head_pitch: float,
        head_yaw: float,
        grid_id: int = -1,
        h_ratio: Optional[float] = None,     # YENİ EKLENDİ
        v_ratio: Optional[float] = None,     # YENİ EKLENDİ
        screen_width: float = 1920,
        screen_height: float = 1080
    ) -> Dict:
        iris_x, iris_y = current_iris
        screen_center_x = screen_width / 2
        screen_center_y = screen_height / 2

        dx_norm = (iris_x - screen_center_x) / screen_width
        dy_norm = (iris_y - screen_center_y) / screen_height
        distance_to_center = math.sqrt((iris_x - screen_center_x) ** 2 + (iris_y - screen_center_y) ** 2)
        angle_to_center = math.atan2(iris_y - screen_center_y, iris_x - screen_center_x)

        record = {
            "iris_x": iris_x,
            "iris_y": iris_y,
            "delta_iris_x": 0.0,
            "delta_iris_y": 0.0,
            "distance_cm": distance_cm,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "delta_head_pitch": 0.0,
            "delta_head_yaw": 0.0,
            "distance_norm": distance_cm / screen_width,
            "iris_x_norm": iris_x / screen_width,
            "iris_y_norm": iris_y / screen_height,
            "angle_id": angle_id,
            "pattern_id": pattern_id,
            "grid_id": grid_id,
            "dx_norm": dx_norm,
            "dy_norm": dy_norm,
            "distance_to_center": distance_to_center,
            "angle_to_center": angle_to_center,
        }

        # Delta hesapla (block başında zaten sıfır olacak)
        if self.prev_iris is not None:
            record["delta_iris_x"] = iris_x - self.prev_iris[0]
            record["delta_iris_y"] = iris_y - self.prev_iris[1]
        if self.prev_pitch is not None:
            record["delta_head_pitch"] = head_pitch - self.prev_pitch
        if self.prev_yaw is not None:
            record["delta_head_yaw"] = head_yaw - self.prev_yaw

        self.prev_iris = current_iris
        self.prev_pitch = head_pitch
        self.prev_yaw = head_yaw

        return record
