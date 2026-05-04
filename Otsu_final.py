import cv2
import os
import json
import numpy as np
import math

def charger_images(dossier):
    images = []
    noms = []
    
    for fichier in os.listdir(dossier):
        if fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            chemin = os.path.join(dossier, fichier)
            img = cv2.imread(chemin)
            
            if img is not None:
                images.append(img)
                noms.append(fichier)
    
    return images, noms

def afficher_image(titre, image, largeur=400):
    h, w = image.shape[:2]
    ratio = largeur / w
    nouvelle_taille = (largeur, int(h * ratio))
    
    image_redim = cv2.resize(image, nouvelle_taille)
    
    cv2.imshow(titre, image_redim)

def pretraitement(image):
    flou = cv2.GaussianBlur(image, (5, 5), 0)
    gris = cv2.cvtColor(flou, cv2.COLOR_BGR2GRAY)
    
    gris = cv2.equalizeHist(gris)
    
    return gris

def seuillage_otsu(image_gris):
    _, thresh = cv2.threshold(
        image_gris, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh

def morphologie(image):
    kernel = np.ones((5,5), np.uint8)

    ouverture = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    fermeture = cv2.morphologyEx(ouverture, cv2.MORPH_CLOSE, kernel)

    dilation = cv2.dilate(fermeture, kernel, iterations=1)

    return dilation

def pipeline(image):
    gris = pretraitement(image)
    
    _, thresh = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    morph = morphologie(thresh)

    contours = trouver_contours(morph)
    contours = filtrer_contours(contours)

    return morph, contours

def trouver_contours(image_binaire):
    contours, _ = cv2.findContours(
        image_binaire,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

def filtrer_contours(contours, aire_min=2000):
    bons = []

    for cnt in contours:
        aire = cv2.contourArea(cnt)
        if aire < aire_min:
            continue

        perimetre = cv2.arcLength(cnt, True)
        if perimetre == 0:
            continue

        x,y,w,h = cv2.boundingRect(cnt)
        ratio = w / h

        if ratio < 0.7 or ratio > 1.3:
            continue

        circularite = 4 * math.pi * aire / (perimetre ** 2)

        if 0.7 < circularite < 1.2:
            bons.append(cnt)

    return bons


def parcourir_dossier(images, noms):
    for i in range(len(images)):
        img = images[i]
        nom = noms[i]

        morph, contours = pipeline(img)

        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, (0,255,0), 2)

        afficher_image(f"Original - {nom}", img)
        afficher_image("Morpho", morph)
        afficher_image("Contours", img_contours)

        print(f"Image {i+1}/{len(images)} : {nom}")
        print("Appuie sur 'n' pour suivant, 'q' pour quitter")

        key = cv2.waitKey(0)

        if key == ord('q'):
            break

        cv2.destroyAllWindows()


def charger_json(fichier):
    with open(fichier, 'r') as f:
        data = json.load(f)

    cercles = []

    for shape in data["shapes"]:
        if shape["shape_type"] != "circle":
            continue

        (cx, cy), (px, py) = shape["points"]

        rayon = math.sqrt((cx - px)**2 + (cy - py)**2)

        cercles.append((cx, cy, rayon))

    return cercles


def contour_vers_cercle(contours):
    cercles = []

    for cnt in contours:
        (x, y), rayon = cv2.minEnclosingCircle(cnt)
        cercles.append((int(x), int(y), int(rayon)))

    return cercles

def distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def matcher(predits, reels, seuil_distance=40, seuil_rayon=30):
    TP = 0
    matched_reels = set()

    for p in predits:
        for i, r in enumerate(reels):
            if i in matched_reels:
                continue

            dist = math.sqrt((p[0]-r[0])**2 + (p[1]-r[1])**2)
            diff_rayon = abs(p[2] - r[2])

            if dist < seuil_distance and diff_rayon < seuil_rayon:
                TP += 1
                matched_reels.add(i)
                break

    FP = len(predits) - TP
    FN = len(reels) - TP

    return TP, FP, FN


def calcul_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def evaluer_image(image, json_path):
    morph, contours = pipeline(image)

    predits = contour_vers_cercle(contours)
    reels = charger_json(json_path)

    TP, FP, FN = matcher(predits, reels)

    precision, recall, f1 = calcul_f1(TP, FP, FN)

    return precision, recall, f1


def evaluer_dataset(images, noms, dossier_json):
    precisions, recalls, f1s = [], [], []

    for i in range(len(images)):
        nom_image = noms[i]
        nom_json = nom_image.replace(".jpg", ".json")

        json_path = os.path.join(dossier_json, nom_json)

        p, r, f1 = evaluer_image(images[i], json_path)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        print(f"{nom_image} → F1 = {f1:.3f}")

    print("\n--- MOYENNE ---")
    print(f"Precision : {sum(precisions)/len(precisions):.3f}")
    print(f"Recall    : {sum(recalls)/len(recalls):.3f}")
    print(f"F1-score  : {sum(f1s)/len(f1s):.3f}")


def dessiner_cercles(image, cercles, couleur=(0,0,255)):
    img = image.copy()
    
    for (x, y, r) in cercles:
        cv2.circle(img, (int(x), int(y)), int(r), couleur, 2)
    
    return img

if __name__ == "__main__":

    dossier_images = "base_images_validation"  
    dossier_json = "validation_annotee"  

    images, noms = charger_images(dossier_images)

    print(f"{len(images)} images chargées")

    parcourir_dossier(images, noms)

    evaluer_dataset(images, noms, dossier_json)