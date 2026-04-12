import cv2
import numpy as np
import os
import json

# CONFIGURATION OPTIMALE (CANNY/CONTOURS) 
C_LOW, C_HIGH = 25, 100
MED_K = 7
GAUS_K = 9
CIRC_LIMIT = 0.85  # 0.6 est assez souple, 1.0 est un cercle parfait

FOLDER_IMAGES = "base_images_validation"
FOLDER_JSON = "validation_annotee"
TARGET_WIDTH = 1024 
THRESHOLD = 0.10  # Tolérance pour le matching (distance centre/rayon)

def get_gt_circles(json_path, ratio):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [(np.array(s['points'][0])*ratio, np.linalg.norm(np.array(s['points'][0])*ratio - np.array(s['points'][1])*ratio)) 
                for s in data['shapes'] if s['shape_type'] == 'circle']
    except: return []

#  INITIALISATION DES COMPTEURS GLOBAUX 
total_found = 0      
total_expected = 0   
total_tp = 0         
total_fp = 0         
total_fn = 0         

filenames = [f for f in os.listdir(FOLDER_IMAGES) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"Démarrage de la visualisation sur {len(filenames)} images...")

for filename in filenames:
    img_orig = cv2.imread(os.path.join(FOLDER_IMAGES, filename))
    if img_orig is None: continue
    
    ratio = TARGET_WIDTH / float(img_orig.shape[1])
    img_display = cv2.resize(img_orig, (TARGET_WIDTH, int(img_orig.shape[0] * ratio)))
    
    #  TRAITEMENT CANNY / CONTOURS
    gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
    processed = cv2.medianBlur(gray, MED_K)
    processed = cv2.GaussianBlur(processed, (GAUS_K, GAUS_K), 0)
    
    # 1. Détection des bords avec Canny
    edges = cv2.Canny(processed, C_LOW, C_HIGH)
    
    # 2. Dilatation pour fermer les contours potentiellement ouverts
    kernel = np.ones((3,3), np.uint8)
    mask_binaire = cv2.dilate(edges, kernel, iterations=1)

    # 3. Recherche des contours sur le masque binaire
    contours, _ = cv2.findContours(mask_binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    det_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Filtre de taille minimale (pour éviter les poussières)
        if area < (TARGET_WIDTH * 0.02)**2 or perimeter == 0:
            continue
            
        # Calcul de la circularité
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        if circularity > CIRC_LIMIT:
            # Transformation du contour en cercle (minEnclosingCircle)
            (x, y), r = cv2.minEnclosingCircle(cnt)
            det_circles.append((int(x), int(y), int(r)))

    # --- MATCHING ET MÉTRIQUES (Idem Hough) ---
    j_path = os.path.join(FOLDER_JSON, os.path.splitext(filename)[0] + ".json")
    gt_data = get_gt_circles(j_path, ratio)
    gt_circles = [(int(c[0][0]), int(c[0][1]), int(c[1])) for c in gt_data]

    img_tp = 0
    matched_gt = [False] * len(gt_circles)
    used_det = [False] * len(det_circles)

    # Copie pour dessiner les résultats
    img_results = img_display.copy()

    # Dessin GT (Rouge)
    for i, (gt_x, gt_y, gt_r) in enumerate(gt_circles):
        cv2.circle(img_results, (gt_x, gt_y), gt_r, (0, 0, 255), 2) # Rouge
        for j, (det_x, det_y, det_r) in enumerate(det_circles):
            if used_det[j]: continue
            dist = np.sqrt((gt_x - det_x)**2 + (gt_y - det_y)**2)
            if dist < (gt_r * THRESHOLD) and abs(gt_r - det_r) < (gt_r * THRESHOLD):
                img_tp += 1
                matched_gt[i] = used_det[j] = True
                break
    
    img_fp = len(det_circles) - img_tp
    img_fn = len(gt_circles) - img_tp

    total_found += len(det_circles)
    total_expected += len(gt_circles)
    total_tp += img_tp
    total_fp += img_fp
    total_fn += img_fn

    # Dessin Détections (Vert)
    for (x, y, r) in det_circles:
        cv2.circle(img_results, (x, y), r, (0, 255, 0), 3) # Vert

    # Overlay texte sur l'image de résultats
    cv2.putText(img_results, f"Ici: {len(det_circles)} det. / {len(gt_circles)} attendues", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # --- AFFICHAGE DES DEUX FENÊTRES ---
    # Fenêtre 1 : Le résultat final avec les cercles
    cv2.imshow("1. Resultats Final (Canny + Contours)", img_results)
    
    # Fenêtre 2 : Ce que 'findContours' voit réellement (Bords dilatés)
    cv2.imshow("2. Masque Binaire (Canny + Dilate)", mask_binaire)
    
    # Positionnement des fenêtres (Optionnel, dépend de ta résolution d'écran)
    cv2.moveWindow("1. Resultats Final (Canny + Contours)", 0, 0)
    cv2.moveWindow("2. Masque Binaire (Canny + Dilate)", TARGET_WIDTH + 10, 0)
    
    print(f"Image: {filename} | Trouvé: {len(det_circles)} | Attendu: {len(gt_circles)}")
    
    if cv2.waitKey(0) == 27: break # Echap pour quitter

cv2.destroyAllWindows()

#  BILAN FINAL 
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "═"*40)
print("       BILAN FINAL (CANNY/CONTOURS)")
print("═"*40)
print(f"PIÈCES TOTALES TROUVÉES  : {total_found}")
print(f"PIÈCES TOTALES ATTENDUES : {total_expected}")
print("-" * 40)
print(f"Vrais Positifs (TP)      : {total_tp}")
print(f"Faux Positifs  (FP)      : {total_fp}")
print(f"Faux Négatifs  (FN)      : {total_fn}")
print("-" * 40)
print(f"PRÉCISION                : {precision:.2%}")
print(f"RAPPEL (RECALL)          : {recall:.2%}")
print(f"F1-SCORE                 : {f1:.2%}")
print("═"*40)