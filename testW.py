import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Veri dosyasını yükle
with open("data/gaze_train_128.box.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Özellikler ve hedef
X_raw = []
y = []
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
screen_center_x = SCREEN_WIDTH / 2
screen_center_y = SCREEN_HEIGHT / 2
screen_diag = (SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2) ** 0.5

for entry in data:
    iris_x = entry["iris_x"]
    iris_y = entry["iris_y"]
    corner_ratios = [
        ((iris_x - ref[0]) ** 2 + (iris_y - ref[1]) ** 2) ** 0.5 / screen_diag
        for ref in [(0, 0), (SCREEN_WIDTH, 0), (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT)]
    ]
    features = [
        iris_x, iris_y, entry["distance_cm"],
        entry["head_pitch"], entry["head_yaw"],
        entry.get("delta_iris_x", 0.0), entry.get("delta_iris_y", 0.0),
        entry.get("delta_head_pitch", 0.0), entry.get("delta_head_yaw", 0.0),
        ((iris_x - screen_center_x) ** 2 + (iris_y - screen_center_y) ** 2) ** 0.5,
        np.arctan2(iris_y - screen_center_y, iris_x - screen_center_x),
        entry.get("h_ratio", 0.0),
        entry.get("v_ratio", 0.0),
        (iris_x - screen_center_x) / SCREEN_WIDTH,
        (iris_y - screen_center_y) / SCREEN_HEIGHT,
        *corner_ratios,
        entry["y1"],
        entry["x1"]
    ]
    X_raw.append(features)
    y.append(entry["y2"])  # m3 target: y2

X_raw = np.array(X_raw)
y = np.array(y)

# Feature isimleri
feature_names = [
    "iris_x", "iris_y", "distance_cm", "head_pitch", "head_yaw",
    "delta_iris_x", "delta_iris_y", "delta_head_pitch", "delta_head_yaw",
    "distance_to_center", "angle_to_center", "h_ratio", "v_ratio",
    "dx_norm", "dy_norm",
    "corner_tl", "corner_tr", "corner_bl", "corner_br",
    "y1 (from m1)", "x1 (from m2)"
]

# Normalize et ve modeli eğit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Feature importance al
importances = model.feature_importances_

# Önem sırasına göre sırala
indices = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in indices]

# Görselleştir
plt.figure(figsize=(10, 6))
plt.title("Feature Importance for m3 (y2 prediction)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), sorted_names, rotation=45, ha="right")
plt.tight_layout()
plt.show()
