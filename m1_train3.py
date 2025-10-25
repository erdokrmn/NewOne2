import json
import numpy as np
import pickle
from collections import Counter
from itertools import product
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

# --- VERİYİ YÜKLE ---
with open("data/gaze_train_128.box.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

X = []
y = []
groups = []

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
screen_center_x = SCREEN_WIDTH / 2
screen_center_y = SCREEN_HEIGHT / 2
screen_diag = distance((0,0), (SCREEN_WIDTH, SCREEN_HEIGHT))

reference_points = [
    (0, 0),                             # sol üst
    (SCREEN_WIDTH, 0),                  # sağ üst
    (0, SCREEN_HEIGHT),                 # sol alt
    (SCREEN_WIDTH, SCREEN_HEIGHT),      # sağ alt
]

for entry in data:
    iris_x = entry["iris_x"]
    iris_y = entry["iris_y"]

    # 4 köşe için oran
    corner_ratios = [
        distance((iris_x, iris_y), ref) / screen_diag
        for ref in reference_points
    ]

    feature = [
        iris_x, iris_y, entry["distance_cm"],
        entry["head_pitch"], entry["head_yaw"],
        entry.get("delta_iris_x", 0.0), entry.get("delta_iris_y", 0.0),
        entry.get("delta_head_pitch", 0.0), entry.get("delta_head_yaw", 0.0),
        ((iris_x - screen_center_x) ** 2 + (iris_y - screen_center_y) ** 2) ** 0.5,         # distance_to_center
        np.arctan2(iris_y - screen_center_y, iris_x - screen_center_x),                    # angle_to_center
        entry.get("h_ratio", 0.0),
        entry.get("v_ratio", 0.0),
        *corner_ratios  # 4 köşe oranı
    ]
    X.append(feature)
    y.append(entry["y1"])              
    groups.append(entry["pattern_id"])

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print(f"Toplam veri: {len(X)}")
print(f"Toplam pattern (blok): {len(set(groups))}")
print("Her pattern başına kayıt (örnek):", Counter(groups).most_common(5))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = GroupKFold(n_splits=5)

model_configs = {
    "RandomForest": {"class": RandomForestClassifier, "params": {"n_estimators": [1000], "max_depth": [20], "n_jobs": [-1], "random_state": [42]}},
    "GradientBoosting": {"class": GradientBoostingClassifier, "params": {"n_estimators": [1000], "learning_rate": [0.1], "max_depth": [3], "random_state": [42]}},
    "LogisticRegression": {"class": LogisticRegression, "params": {"max_iter": [1000], "solver": ["lbfgs"], "random_state": [42]}},
    "KNN": {"class": KNeighborsClassifier, "params": {"n_neighbors": [50], "n_jobs": [-1], "leaf_size": [100]}}
}
if XGBClassifier:
    model_configs["XGBoost"] = {"class": XGBClassifier, "params": {"n_estimators": [1000], "max_depth": [8], "learning_rate": [0.005], "eval_metric": ["mlogloss"], "verbosity": [0],  "random_state": [42]}}
if CatBoostClassifier:
    model_configs["CatBoost"] = {"class": CatBoostClassifier, "params": {"iterations": [1000], "depth": [8], "learning_rate": [0.005], "verbose": [0], "random_state": [42]}}
if LGBMClassifier:
    model_configs["LightGBM"] = {"class": LGBMClassifier, "params": {"n_estimators": [1000], "max_depth": [8], "learning_rate": [0.005], "random_state": [42], "verbosity": [0]}}


def generate_param_combinations(param_grid):
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        yield dict(zip(keys, values))

best_score = -1
best_model = None
best_name = ""
best_params = {}

for model_name, cfg in model_configs.items():
    for param_set in generate_param_combinations(cfg["params"]):
        model = cfg["class"](**param_set)
        scores = cross_val_score(model, X_scaled, y, cv=cv, groups=groups)
        mean_acc = scores.mean()
        if mean_acc > best_score:
            best_score = mean_acc
            best_model = model
            best_name = model_name
            best_params = param_set

print(f"\n🔹 En iyi model: {best_name} {best_params} ({best_score:.3f})")
best_model.fit(X_scaled, y)
with open("data/gaze_grid_m1.pkl", "wb") as f:
    pickle.dump({"model": best_model, "scaler": scaler}, f)
print("✅ gaze_grid_m1.pkl olarak kaydedildi.")

# ---- FEATURE IMPORTANCE ----
feature_names = [
    "iris_x", "iris_y", "distance_cm", "head_pitch", "head_yaw",
    "delta_iris_x", "delta_iris_y", "delta_head_pitch", "delta_head_yaw",
    "distance_to_center", "angle_to_center", "h_ratio", "v_ratio",
    "ratio_left_top", "ratio_right_top", "ratio_left_bottom", "ratio_right_bottom"
]
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    print("\n--- Özellik Önemleri (Feature Importance) ---")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"{fname:20s} : {imp:.4f}")
else:
    print("Bu modelde feature_importance özelliği yok.")
