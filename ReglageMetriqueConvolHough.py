import cv2
import numpy as np
import os
import json
from itertools import product
from concurrent.futures import ProcessPoolExecutor


# --- CONFIGURATION ---
FOLDER_IMAGES = "base_images_validation"
FOLDER_JSON = "validation_annotee"
TARGET_WIDTH = 1024
THRESHOLD = 0.10


def get_gt_circles(json_path, ratio):

    with open(json_path, "r", encoding="utf-8") as f:

        data = json.load(f)

    circles = []

    for shape in data["shapes"]:

        if shape["shape_type"] == "circle":

            p1, p2 = shape["points"]

            center = np.array(p1) * ratio

            edge = np.array(p2) * ratio

            radius = np.linalg.norm(center - edge)

            circles.append((center[0], center[1], radius))

    return circles


def load_all_data():
    """Pré-charge les images et les annotations en mémoire."""

    preloaded = []

    filenames = [
        f
        for f in os.listdir(FOLDER_IMAGES)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"Chargement de {len(filenames)} images en mémoire...")

    for filename in filenames:

        img = cv2.imread(os.path.join(FOLDER_IMAGES, filename))

        if img is None:
            continue

        h, w = img.shape[:2]

        ratio = TARGET_WIDTH / float(w)

        img_rs = cv2.resize(img, (TARGET_WIDTH, int(h * ratio)))

        gray = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)

        json_path = os.path.join(FOLDER_JSON, os.path.splitext(filename)[0] + ".json")

        if os.path.exists(json_path):

            gt = get_gt_circles(json_path, ratio)

            preloaded.append((gray, gt))

    return preloaded


def evaluate_one_config(params, data_list):
    """Fonction exécutée par chaque cœur du processeur."""

    p1, p2, dp, md, mk, gk = params

    tp, fp, fn = 0, 0, 0

    for gray, gt_circles in data_list:

        # Prétraitement

        processed = cv2.medianBlur(gray, mk)

        processed = cv2.GaussianBlur(processed, (gk, gk), 0)

        gray_eq = cv2.createCLAHE(clipLimit=2.0).apply(processed)

        # Détection

        detected = cv2.HoughCircles(
            gray_eq,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=md,
            param1=p1,
            param2=p2,
            minRadius=int(TARGET_WIDTH * 0.04),
            maxRadius=int(TARGET_WIDTH * 0.25),
        )

        det_circles = (
            np.around(detected[0, :]).astype(int) if detected is not None else []
        )

        # Matching

        matched_gt = [False] * len(gt_circles)

        used_det = [False] * len(det_circles)
        for i, (gt_x, gt_y, gt_r) in enumerate(gt_circles):
            for j, (det_x, det_y, det_r) in enumerate(det_circles):
                if used_det[j]:
                    continue
                dist = np.sqrt((gt_x - det_x) ** 2 + (gt_y - det_y) ** 2)
                if dist < (gt_r * THRESHOLD) and abs(gt_r - det_r) < (gt_r * THRESHOLD):
                    tp += 1
                    matched_gt[i] = used_det[j] = True
                    break
        fn += matched_gt.count(False)
        fp += used_det.count(False)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return (params, f1, prec, rec)


if __name__ == "__main__":
    #LISTES DES PARAMÈTRES 
    p1_list = [120, 140]
    p2_list = [70, 75, 80]
    dp_list = [1.1, 1.15, 1.2, 1.25]
    dist_list = [40, 60, 80, 100]
    median_k_list = [9, 11, 13]
    gaussian_k_list = [11, 13, 15]
    all_data = load_all_data()
    combinations = list(
        product(p1_list, p2_list, dp_list, dist_list, median_k_list, gaussian_k_list)
    )

    total = len(combinations)
    print(f"DÉMARRAGE : {total} combinaisons sur {len(all_data)} images.")
    print("Utilisation du multi-processing (tous les cœurs)...")
    best_f1 = 0
    best_params = None
    with ProcessPoolExecutor(max_workers=2) as executor:
        # On lance les tâches. On utilise chunksize pour plus d'efficacité
        future_results = [
            executor.submit(evaluate_one_config, c, all_data) for c in combinations
        ]
        for i, future in enumerate(future_results):
            params, f1, prec, rec = future.result()
            # Affichage de suivi (toutes les 10 combis pour ne pas saturer la console)
            if i % 10 == 0 or f1 > best_f1:
                print(
                    f"[{i}/{total}] F1 Actuel: {f1:.2%} | P1:{params[0]} P2:{params[1]} dp:{params[2]}"
                )
            if f1 > best_f1:
                best_f1 = f1
                best_params = (params, f1, prec, rec)
                print(f"       NOUVEAU MEILLEUR : {best_f1:.2%}")
    # --- RÉSULTATS FINAUX ---
    if best_params:
        p, f, pr, re = best_params
        print("\n" + "═" * 50)
        print("  CONFIGURATION OPTIMALE TERMINÉE")
        print(
            f" Paramètres : P1={p[0]}, P2={p[1]}, dp={p[2]}, dist={p[3]}, med={p[4]}, gaus={p[5]}"
        )
        print(f" F1: {f:.2%} | Précision: {pr:.2%} | Rappel: {re:.2%}")
        print("═" * 50)
