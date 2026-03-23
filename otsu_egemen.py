import cv2
import numpy as np
import math

def compter_pieces_avance(image_path, min_area=1500, show_steps=True):
    # 1. Chargement de l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur : Image introuvable -> {image_path}")
        return
    
    output = img.copy()
    
    # 2. Prétraitement : Conversion en nuances de gris et floutage
    # Le flou médian est idéal pour supprimer les détails (gravures) sur les pièces
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flou = cv2.medianBlur(gris, 7)

    # 3. Seuillage d'Otsu (Binarisation automatique)
    # On utilise THRESH_BINARY_INV car les pièces sont généralement plus sombres que le fond
    _, seuil = cv2.threshold(flou, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Opérations morphologiques (Fermer les trous et supprimer le bruit)
    kernel = np.ones((5,5), np.uint8)
    fermeture = cv2.morphologyEx(seuil, cv2.MORPH_CLOSE, kernel, iterations=2)
    ouverture = cv2.morphologyEx(fermeture, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Détection des contours
    contours, _ = cv2.findContours(ouverture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nombre_pieces = 0
    for cnt in contours:
        aire = cv2.contourArea(cnt)
        perimetre = cv2.arcLength(cnt, True)
        
        if perimetre == 0 or aire < min_area:
            continue

        # Calcul de la circularité : 1.0 correspond à un cercle parfait
        circularite = (4 * math.pi * aire) / (perimetre * perimetre)
        
        # Un intervalle entre 0.7 et 1.2 est idéal pour les pièces de monnaie
        if 0.7 < circularite < 1.2:
            nombre_pieces += 1
            
            # Visualisation : Dessiner le cercle englobant minimum
            (x, y), rayon = cv2.minEnclosingCircle(cnt)
            centre = (int(x), int(y))
            cv2.circle(output, centre, int(rayon), (0, 255, 0), 3)
            cv2.putText(output, f"Piece #{nombre_pieces}", (int(x)-20, int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    print(f"Total des pièces détectées : {nombre_pieces}")

    # Affichage des étapes si activé
    if show_steps:
        cv2.imshow("1. Gris et Flou", flou)
        cv2.imshow("2. Seuillage d'Otsu", seuil)
        cv2.imshow("3. Masque Nettoye", ouverture)
        cv2.imshow("4. Resultat Final", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Remplacez le chemin par votre image de validation
    chemin_image = "base_images_validation/WhatsApp Image 2026-02-11 at 15.38.17 (2).jpeg"
    compter_pieces_avance(chemin_image)
