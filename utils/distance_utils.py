import cv2
import mediapipe as mp
import json
import os

model_path = os.path.join(os.path.dirname(__file__), "..", "data", "z_model.json")


model_path = os.path.abspath(model_path)

if os.path.exists(model_path):
    with open(model_path, "r") as f:
        calibration = json.load(f)
        a = calibration["a"]
        b = calibration["b"]
else:
    print("⚠️ Kalibrasyon modeli bulunamadı! Varsayılan değerler kullanılacak (a=1, b=0).")
    a, b = 1.0, 0.0
    
# MediaPipe face detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

def estimate_distance_cm(frame):
    """
    MediaPipe ile yüz genişliğini tespit eder ve mesafeyi tahmin eder.
    """
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if not results.detections:
        return 0.0

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    
    face_width_px = bbox.width * width

    

    if face_width_px <= 1:
        return 0.0

    distance_cm = a / face_width_px + b
    return round(distance_cm, 2)
