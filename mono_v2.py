import os
import cv2
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------- Настройки -----------------
IMAGE_FOLDER = r"MH_02_easy\mav0"
EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

N_FEATURES = 1000
RATIO_TEST = 0.75
MAX_MATCHES_KEEP = 500

# ----------------- Загрузка калибровки -----------------
def find_calib_file(folder):
    for name in ['sensor.yaml','sensor.yml','camera.yaml','cam0.yaml','calib.yaml','calib.txt']:
        for p in Path(folder).rglob(name):
            return str(p)
    return None

def parse_calib_euroc(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    intr = data.get('intrinsics', None)
    dist = data.get('distortion_coefficients', None)
    if intr is None or len(intr) < 4:
        raise ValueError("не нашёл ключ intrinsics")
    fx, fy, cx, cy = intr[:4]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
    distc = np.zeros(5)
    if dist is not None:
        d = np.array(dist, dtype=float).ravel()
        distc[:len(d)] = d[:5]
    return K, distc

# ----------------- Загрузка изображений -----------------
def load_images(folder):
    files = sorted([str(p) for ext in EXTS for p in Path(folder).rglob(f'*{ext}')])
    imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
    return [img for img in imgs if img is not None]

# ----------------- Монокулярная VO -----------------
def run_mono_vo(image_folder):
    calib_path = find_calib_file(image_folder)
    if not calib_path:
        raise FileNotFoundError("Файл калибровки не найден")
    K, dist = parse_calib_euroc(calib_path)
    print("Используем полную калибровку камеры:\nK=\n", K, "\nDist=", dist)

    images = load_images(image_folder)
    if len(images) == 0:
        print("Нет изображений")
        return

    # ----------------- ORB -----------------
    orb = cv2.ORB_create(N_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    trajectory = []
    pose = np.eye(4)

    # ----------------- Инициализация -----------------
    prev_img = cv2.undistort(images[0], K, dist)
    kp_prev, des_prev = orb.detectAndCompute(prev_img, None)
    trajectory.append(pose[:3,3].copy())

    # ----------------- Основной цикл -----------------
    for idx, img in enumerate(images[1:], 1):
        cur_img = cv2.undistort(img, K, dist)
        kp_cur, des_cur = orb.detectAndCompute(cur_img, None)
        if des_prev is None or des_cur is None:
            kp_prev, des_prev = kp_cur, des_cur
            trajectory.append(pose[:3,3].copy())
            continue

        matches = bf.knnMatch(des_prev, des_cur, k=2)
        good = [m for m,n in matches if m.distance < RATIO_TEST*n.distance][:MAX_MATCHES_KEEP]
        if len(good) < 8:
            kp_prev, des_prev = kp_cur, des_cur
            trajectory.append(pose[:3,3].copy())
            continue

        pts_prev = np.array([kp_prev[m.queryIdx].pt for m in good], dtype=np.float32)
        pts_cur  = np.array([kp_cur[m.trainIdx].pt for m in good], dtype=np.float32)

        E, mask = cv2.findEssentialMat(pts_cur, pts_prev, K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            kp_prev, des_prev = kp_cur, des_cur
            trajectory.append(pose[:3,3].copy())
            continue

        _, R, t, _ = cv2.recoverPose(E, pts_cur, pts_prev, K, mask=mask)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.ravel()
        pose = pose @ np.linalg.inv(T)
        trajectory.append(pose[:3,3].copy())

        kp_prev, des_prev = kp_cur, des_cur

        if idx % 50 == 0:
            print(f"[{idx}/{len(images)}] кадров обработано")

    trajectory = np.array(trajectory)
    np.save('trajectory.npy', trajectory)
    print("Сохранено trajectory.npy")

    # ----------------- Визуализация -----------------
    plt.figure(figsize=(8,6))
    plt.plot(trajectory[:,0], trajectory[:,2], '-o', markersize=2)
    plt.title("Вычисленная траектория (X по сравнению с Z) — полная калибровка камеры cam0")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis('equal')
    plt.grid()
    plt.show()
    print("✅ Done.")

if __name__ == '__main__':
    run_mono_vo(IMAGE_FOLDER)
