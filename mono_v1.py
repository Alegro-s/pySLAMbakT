import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- Пользовательские настройки -----------------
IMAGE_FOLDER = r"MH_02_easy\mav0"
EXTS = ('.png', '.jpg', '.jpeg', '.bmp')

N_FEATURES = 2000
RATIO_TEST = 0.75
MAX_MATCHES_KEEP = 1500

# ---------- Разбор калибровки ----------
def find_calib_file(folder):
    candidates = [
        'sensor.yaml', 'sensor.yml', 'camera.yaml', 'cam0.yaml', 'calib.yaml', 'calib.txt'
    ]
    for c in candidates:
        for p in Path(folder).rglob(c):
            return str(p)
    return None


def parse_calib_euroc(path):
    if path is None or not os.path.exists(path):
        return None, None

    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        intr = data.get('внутренние параметры камеры', None)
        dist = data.get('Параметр искажения', None)

        if intr is None or len(intr) < 4:
            print("Файл калибровки найден.")
            return None, None

        fx, fy, cx, cy = intr[:4]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=float)

        distc = np.zeros(5)
        if dist is not None and len(dist) > 0:
            d = np.array(dist, dtype=float).ravel()
            distc[:len(d)] = d[:5]  # копируем максимум 5 коэффициентов
        print(f"Калибровка:\nK=\n{K}\nDist={distc}")
        return K, distc

    except Exception as e:
        print("Не удалось разобрать файл калибровки:", e)
        return None, None


# ---------- Загрузка картинок ----------
def load_images_from_folder(folder):
    files = []
    for ext in EXTS:
        files += [str(p) for p in Path(folder).rglob(f'*{ext}')]
    files = sorted(files)
    imgs = []
    for p in files:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            imgs.append((p, img))
    return imgs


# ---------- Визуализация ----------
def run_mono_vo(image_folder):
    calib_path = find_calib_file(image_folder)
    if calib_path:
        print("Не найден файл калибровки:", calib_path)
    K, dist = parse_calib_euroc(calib_path)

    images = load_images_from_folder(image_folder)
    if len(images) == 0:
        print("Нет изображений.")
        return

    if K is None:
        h, w = images[0][1].shape
        f = 0.8 * w
        K = np.array([[f, 0, w/2.0],
                      [0, f, h/2.0],
                      [0, 0, 1]], dtype=float)
        dist = np.zeros(5)
        print("⚠️ Калибровка не найдена:\n", K)
    else:
        print("✅ Пользовательская калибровачный файл в папке.")

    need_undistort = np.any(np.abs(dist) > 1e-9)

    orb = cv2.ORB_create(N_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    pose = np.eye(4, dtype=float)
    trajectory = [pose.copy()]
    xs, zs = [], []

    prev_path, prev_img = images[0]
    if need_undistort:
        prev_img = cv2.undistort(prev_img, K, dist)
    kp_prev, des_prev = orb.detectAndCompute(prev_img, None)
    print(f"Загрузка {len(images)} Фрейм. Starting VO...")

    for idx in range(1, len(images)):
        path_cur, img_cur = images[idx]
        if need_undistort:
            img_cur = cv2.undistort(img_cur, K, dist)
        kp_cur, des_cur = orb.detectAndCompute(img_cur, None)
        if des_prev is None or des_cur is None:
            kp_prev, des_prev = kp_cur, des_cur
            continue

        matches = bf.knnMatch(des_prev, des_cur, k=2)
        good = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < RATIO_TEST * n.distance:
                good.append(m)

        if len(good) < 8:
            kp_prev, des_prev = kp_cur, des_cur
            continue

        good = sorted(good, key=lambda x: x.distance)[:MAX_MATCHES_KEEP]
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
        pts_cur  = np.float32([kp_cur[m.trainIdx].pt for m in good])

        E, maskE = cv2.findEssentialMat(pts_cur, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            kp_prev, des_prev = kp_cur, des_cur
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts_cur, pts_prev, K, mask=maskE)

        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.ravel()

        pose = pose @ np.linalg.inv(T)
        trajectory.append(pose.copy())
        xs.append(pose[0,3])
        zs.append(pose[2,3])

        if idx % 50 == 0 or idx == len(images)-1:
            print(f"[{idx}/{len(images)}] обработано, {len(good)} точки")

        kp_prev, des_prev = kp_cur, des_cur

    np.save('trajectory.npy', np.stack(trajectory, axis=0))
    print("Сохранение trajectory.npy")

    if len(xs) > 0:
        plt.figure(figsize=(8,6))
        plt.plot(xs, zs, '-o', markersize=2)
        plt.title("Вычисленная траектория (X по сравнению с Z) — калибровка камеры cam0 из EuRoC")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.axis('equal')
        plt.grid()
        plt.savefig('trajectory.png', dpi=200)
        plt.show()
        print("Сохранение trajectory.png")

    print("✅ Done.")

if __name__ == '__main__':
    run_mono_vo(IMAGE_FOLDER)