import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh başlat
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1
)

def get_head_pose(frame):
    """
    Verilen bir kamera görüntüsünden kafa yönünü (pitch ve yaw açıları) derece cinsinden hesaplar.
    
    Dönüş:
        pitch: Başın yukarı-aşağı eğimi (derece)
        yaw: Başın sağa-sola dönüşü (derece)
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None, None  # Yüz algılanamadıysa None döner

    landmarks = results.multi_face_landmarks[0].landmark
    image_height, image_width, _ = frame.shape

    # Belirli landmark noktalarını seçiyoruz
    indices = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye": 263,
        "right_eye": 33,
        "left_mouth": 287,
        "right_mouth": 57
    }

    face_2d = []
    face_3d = []

    for name, idx in indices.items():
        lm = landmarks[idx]
        x, y = int(lm.x * image_width), int(lm.y * image_height)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z * 3000])  # Z koordinatı ölçeklenerek ekleniyor

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Kamera matrisini tanımla
    focal_length = image_width
    cam_matrix = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ])

    dist_coeffs = np.zeros((4, 1))  # Lens distorsiyonu yok

    # solvePnP ile kafa pozisyonu vektörü hesaplanıyor
    success, rotation_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
    if not success:
        return None, None

    # Rotasyon matrisinden Euler açıları elde edilir
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch = euler_angles[0][0]  # yukarı-aşağı
    yaw = euler_angles[1][0]    # sağa-sola
    return round(pitch, 2), round(yaw, 2)
