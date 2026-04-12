import cv2
import numpy as np
import os
import json

# --- CONFIGURATION DES PARAMÈTRES GAGNANTS ---
P1, P2 = 140, 75
DP = 1.25
DIST = 100
MED_K = 9
GAUS_K = 11

FOLDER_IMAGES = "base_images_validation"
FOLDER_JSON = "validation_annotee"
TARGET_WIDTH = 1024 
THRESHOLD = 0.10

def get_gt_circles(json_path, ratio):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [(np.array(s['points'][0])*ratio, np.linalg.norm(np.array(s['points'][0])*ratio - np.array(s['points'][1])*ratio)) 
                for s in data['shapes'] if s['shape_type'] == 'circle']
    except: return []

# --- INITIALISATION DES COMPTEURS GLOBAUX ---
total_found = 0      # Total de cercles détectés par l'algorithme
total_expected = 0   # Total de cercles annotés (vérité terrain)
total_tp = 0         # Vrais Positifs (bonnes détections)
total_fp = 0         # Faux Positifs (erreurs/fantômes)
total_fn = 0         # Faux Négatifs (pièces oubliées)

filenames = [f for f in os.listdir(FOLDER_IMAGES) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for filename in filenames:
    img_orig = cv2.imread(os.path.join(FOLDER_IMAGES, filename))
    if img_orig is None: continue
    
    ratio = TARGET_WIDTH / float(img_orig.shape[1])
    img_display = cv2.resize(img_orig, (TARGET_WIDTH, int(img_orig.shape[0] * ratio)))
    
    gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
    processed = cv2.medianBlur(gray, MED_K)
    processed = cv2.GaussianBlur(processed, (GAUS_K, GAUS_K), 0)
    gray_eq = cv2.createCLAHE(clipLimit=2.0).apply(processed)

    detected = cv2.HoughCircles(gray_eq, cv2.HOUGH_GRADIENT, dp=DP, minDist=DIST,
                                param1=P1, param2=P2, 
                                minRadius=int(TARGET_WIDTH*0.04), maxRadius=int(TARGET_WIDTH*0.25))
    
    det_circles = np.around(detected[0, :]).astype(int) if detected is not None else []

    j_path = os.path.join(FOLDER_JSON, os.path.splitext(filename)[0] + ".json")
    gt_data = get_gt_circles(j_path, ratio)
    gt_circles = [(int(c[0][0]), int(c[0][1]), int(c[1])) for c in gt_data]

    # Métriques de l'image actuelle
    img_tp = 0
    matched_gt = [False] * len(gt_circles)
    used_det = [False] * len(det_circles)

    for i, (gt_x, gt_y, gt_r) in enumerate(gt_circles):
        cv2.circle(img_display, (gt_x, gt_y), gt_r, (0, 0, 255), 2) # GT en Rouge
        for j, (det_x, det_y, det_r) in enumerate(det_circles):
            if used_det[j]: continue
            dist = np.sqrt((gt_x - det_x)**2 + (gt_y - det_y)**2)
            if dist < (gt_r * THRESHOLD) and abs(gt_r - det_r) < (gt_r * THRESHOLD):
                img_tp += 1
                matched_gt[i] = used_det[j] = True
                break
    
    img_fp = len(det_circles) - img_tp
    img_fn = len(gt_circles) - img_tp

    # --- MISE À JOUR DES COMPTEURS GLOBAUX ---
    total_found += len(det_circles)
    total_expected += len(gt_circles)
    total_tp += img_tp
    total_fp += img_fp
    total_fn += img_fn

    for (x, y, r) in det_circles:
        cv2.circle(img_display, (x, y), r, (0, 255, 0), 3) # Détection en Vert

    # Affichage overlay
    cv2.putText(img_display, f"Ici: {len(det_circles)} det. / {len(gt_circles)} attendues", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Visualisation", img_display)
    print(f"Image: {filename} | Trouvé: {len(det_circles)} | Attendu: {len(gt_circles)}")
    
    if cv2.waitKey(0) == 27: break

cv2.destroyAllWindows()

# --- RÉCAPITULATIF FINAL DANS LA CONSOLE ---
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*40)
print("       BILAN DE L'ALGORITHME")
print("="*40)
print(f"PIÈCES TOTALES TROUVÉES  : {total_found}")
print(f"PIÈCES TOTALES ATTENDUES : {total_expected}")
print("-" * 40)
print(f"Vrais Positifs (TP)      : {total_tp} (Correctement identifiées)")
print(f"Faux Positifs  (FP)      : {total_fp} (Erreurs de détection)")
print(f"Faux Négatifs  (FN)      : {total_fn} (Pièces manquées)")
print("-" * 40)
print(f"PRÉCISION                : {precision:.2%}")
print(f"RAPPEL (RECALL)          : {recall:.2%}")
print(f"F1-SCORE                 : {f1:.2%}")
print("="*40)