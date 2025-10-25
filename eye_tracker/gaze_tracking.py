from __future__ import division
import os
import cv2
import dlib
import numpy as np
from eye_tracker.eye import Eye
from eye_tracker.calibration import Calibration

# ----------- YARDIMCI FONKSİYONLAR (SINIF DIŞI) -----------
def _find_pupil_in_eye(eye_img):
    if eye_img is None or eye_img.size == 0:
        return None
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

def _normalized_pupil_coords(pupil_xy, bbox):
    if pupil_xy is None or bbox is None:
        return 0.5, 0.5
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    rel_x = pupil_xy[0] / w if w != 0 else 0.5
    rel_y = pupil_xy[1] / h if h != 0 else 0.5
    rel_x = min(max(rel_x, 0.0), 1.0)
    rel_y = min(max(rel_y, 0.0), 1.0)
    return rel_x, rel_y

# ----------- GazeTracking Sınıfı -----------
class GazeTracking(object):
    """
    Eye-tracking ana sınıfı
    """
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        self._face_detector = dlib.get_frontal_face_detector()
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame_gray)
        try:
            landmarks = self._predictor(frame_gray, faces[0])
            self.eye_left = Eye(frame_gray, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame_gray, landmarks, 1, self.calibration)
        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """
        0.0: sağ, 1.0: sol (göz kutusunda normalize pupil yeri)
        """
        if self.pupils_located and self.frame is not None:
            left_bbox = self.eye_left.bbox()
            right_bbox = self.eye_right.bbox()
            frame = self.frame
            left_eye_img = frame[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
            right_eye_img = frame[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]
            left_pupil = _find_pupil_in_eye(left_eye_img) if left_eye_img.size > 0 else None
            right_pupil = _find_pupil_in_eye(right_eye_img) if right_eye_img.size > 0 else None
            h_left, _ = _normalized_pupil_coords(left_pupil, left_bbox)
            h_right, _ = _normalized_pupil_coords(right_pupil, right_bbox)
            return (h_left + h_right) / 2
        return 0.5

    def vertical_ratio(self):
        """
        0.0: üst, 1.0: alt (göz kutusunda normalize pupil yeri)
        """
        if self.pupils_located and self.frame is not None:
            left_bbox = self.eye_left.bbox()
            right_bbox = self.eye_right.bbox()
            frame = self.frame
            left_eye_img = frame[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
            right_eye_img = frame[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]
            left_pupil = _find_pupil_in_eye(left_eye_img) if left_eye_img.size > 0 else None
            right_pupil = _find_pupil_in_eye(right_eye_img) if right_eye_img.size > 0 else None
            _, v_left = _normalized_pupil_coords(left_pupil, left_bbox)
            _, v_right = _normalized_pupil_coords(right_pupil, right_bbox)
            return (v_left + v_right) / 2
        return 0.5

    def is_right(self):
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 6

    def annotated_frame(self):
        frame = self.frame.copy()
        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
        return frame
