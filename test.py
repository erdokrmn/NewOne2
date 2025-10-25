import json
import numpy as np
import matplotlib.pyplot as plt

# Veriyi yükle
with open("data/gaze_train_128.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Delta değerleri topla
delta_iris_x = [entry.get("delta_iris_x", 0.0) for entry in data]
delta_iris_y = [entry.get("delta_iris_y", 0.0) for entry in data]
delta_head_pitch = [entry.get("delta_head_pitch", 0.0) for entry in data]
delta_head_yaw = [entry.get("delta_head_yaw", 0.0) for entry in data]

# Histogram çiz (her 10 aralığında bin)
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

axs[0, 0].hist(delta_iris_x, bins=np.arange(-100, 130, 10), edgecolor='black')
axs[0, 0].set_title('delta_iris_x')
axs[0, 0].set_xlabel('Değer')
axs[0, 0].set_ylabel('Frekans')

axs[0, 1].hist(delta_iris_y, bins=np.arange(-80, 110, 10), edgecolor='black')
axs[0, 1].set_title('delta_iris_y')
axs[0, 1].set_xlabel('Değer')
axs[0, 1].set_ylabel('Frekans')

axs[1, 0].hist(delta_head_pitch, bins=np.arange(-15, 15, 2), edgecolor='black')
axs[1, 0].set_title('delta_head_pitch')
axs[1, 0].set_xlabel('Değer')
axs[1, 0].set_ylabel('Frekans')

axs[1, 1].hist(delta_head_yaw, bins=np.arange(-35, 35, 5), edgecolor='black')
axs[1, 1].set_title('delta_head_yaw')
axs[1, 1].set_xlabel('Değer')
axs[1, 1].set_ylabel('Frekans')

plt.tight_layout()
plt.show()
