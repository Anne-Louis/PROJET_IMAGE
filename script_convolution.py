from pathlib import Path
import cv2
from scipy import signal
import numpy as np

# Ta matrice de convolution (Laplacien)
matrice_convolution = [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]

# Chargement de l'image
path_img = "1000016160.jpg"
image = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image introuvable !")
else:
    # 1. Convolution
    res = signal.convolve2d(image, matrice_convolution, mode='same')

    # 2. Conversion en 8-bit (uint8) pour OpenCV
    res_display = np.absolute(res).astype(np.uint8)

    # 3. Détection de cercles
    Hough_test = cv2.HoughCircles(res_display, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                  param1=25, param2=50, minRadius=125, maxRadius=150)

    # 4. Dessin sur l'image de résultat
    if Hough_test is not None:
        Hough_test = np.uint16(np.around(Hough_test))
        for i in Hough_test[0, :]:
            # On dessine sur res_display
            cv2.circle(res_display, (i[0], i[1]), i[2], (255, 255, 255), 2)
            cv2.circle(res_display, (i[0], i[1]), 2, (255, 255, 255), 3)
    else:
        print("Aucun cercle détecté.")

    # 5. Affichage
    cv2.imshow("test", res_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()