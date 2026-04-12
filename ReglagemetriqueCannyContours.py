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
    try:
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
    except: return []

def load_all_data():
    preloaded = []
    filenames = [f for f in os.listdir(FOLDER_IMAGES) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"Chargement de {len(filenames)} images...")
    for filename in filenames:
        img = cv2.imread(os.path.join(FOLDER_IMAGES, filename))
        if img is None: continue
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
    c_low, c_high, mk, gk, circ_limit, dil_iter = params
    tp, fp, fn = 0, 0, 0

    for gray, gt_circles in data_list:
        # 1. Prétraitement
        processed = cv2.medianBlur(gray, mk)
        processed = cv2.GaussianBlur(processed, (gk, gk), 0)
        
        # 2. Canny
        edges = cv2.Canny(processed, c_low, c_high)
        
        # 3. Dilatation (Aide à boucher les petits trous avant le Hull)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=dil_iter)

        # 4. Détection des contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        det_circles = []
        for cnt in contours:
            # --- FERMETURE DU "C" VIA CONVEX HULL ---
            # On crée l'enveloppe convexe du contour
            hull = cv2.convexHull(cnt)
            
            # On calcule l'aire et le périmètre de l'enveloppe, pas du contour brut
            area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            
            if area < (TARGET_WIDTH * 0.03)**2 or area > (TARGET_WIDTH * 0.3)**2:
                continue
            if perimeter == 0: continue
            
            # Circularité sur l'enveloppe (plus stable pour les formes fermées artificiellement)
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            
            if circularity > circ_limit:
                (x, y), radius = cv2.minEnclosingCircle(hull)
                det_circles.append((x, y, radius))

        # 5. Matching
        matched_gt = [False] * len(gt_circles)
        used_det = [False] * len(det_circles)
        for i, (gt_x, gt_y, gt_r) in enumerate(gt_circles):
            for j, (det_x, det_y, det_r) in enumerate(det_circles):
                if used_det[j]: continue
                dist = np.sqrt((gt_x - det_x)**2 + (gt_y - det_y)**2)
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
    # --- GRILLE DE PARAMÈTRES ADAPTÉE ---
    canny_low = [10, 20, 30, 40, 50] 
    canny_high = [100, 120, 150, 200]
    median_k = [7, 11, 15]
    gauss_k = [9, 13, 17]
    # Avec le Convex Hull, on peut être plus exigeant sur la circularité
    circularity_min = [0.6,0.65,0.7, 0.75, 0.8, 0.85] 
    dilate_iterations = [1, 2, 3] # Le Hull fait déjà le gros du travail de fermeture

    all_data = load_all_data()
    combinations = list(product(canny_low, canny_high, median_k, gauss_k, circularity_min, dilate_iterations))
    
    total = len(combinations)
    print(f"DÉMARRAGE : {total} combinaisons (Mode Convex Hull).")

    best_f1 = 0
    best_params = None

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_results = [executor.submit(evaluate_one_config, c, all_data) for c in combinations]
        
        for i, future in enumerate(future_results):
            params, f1, prec, rec = future.result()
            if i % 10 == 0 or f1 > best_f1:
                print(f"[{i}/{total}] F1: {f1:.2%} | Low:{params[0]} High:{params[1]} | Circ:{params[4]}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = (params, f1, prec, rec)
                print(f"         NOUVEAU MEILLEUR : {best_f1:.2%}")

    if best_params:
        p, f, pr, re = best_params
        print("\n" + "═"*50)
        print("  CONFIGURATION OPTIMALE (CONVEX HULL)")
        print(f" Canny: {p[0]}/{p[1]} | Flous: M{p[2]}/G{p[3]}")
        print(f" Circ: {p[4]} | Dilatation: {p[5]}")
        print(f" F1: {f:.2%} | Précision: {pr:.2%} | Rappel: {re:.2%}")
        print("═"*50)