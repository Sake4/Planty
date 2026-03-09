import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def process_taille(filepath, output_path):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lower = np.array([60, 60, 60])
    higher = np.array([250, 250, 250])
    masque = cv2.inRange(image, lower, higher)

    contours_masque, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    taille = 0
    if contours_masque:
        c = max(contours_masque, key=cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(c)
        cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 5)
        taille = h1
        title = f'Taille: {h1}'
    else:
        title = 'Aucun contour trouvé'

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title, fontsize=12, color='blue')
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return {'taille': taille}

def process_couleur_verte(filepath, output_path):
    # Charger et convertir l'image
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Étape 1 : Détection des parties vertes
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 40, 40])
    higher_green = np.array([85, 255, 255])
    masque_verte = cv2.inRange(hsv, lower_green, higher_green)
    partie_verte = cv2.bitwise_and(image, image, mask=masque_verte)

    # Étape 2 : Compter les feuilles
    contours_feuilles, _ = cv2.findContours(masque_verte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nombre_feuilles_vertes = len(contours_feuilles)
    print(f"Nombre de feuilles détectées : {nombre_feuilles_vertes}")

    # Dessiner les contours des feuilles
    image_contours = cv2.drawContours(image, contours_feuilles, -1, (0, 255, 0), 2)
    plt.figure("Graph Final", figsize=(20, 4))
    plt.subplot(1, 3, 1), plt.imshow(masque_verte)
    plt.subplot(1, 3, 2), plt.imshow(partie_verte)
    plt.title("Parties vertes détectées")
    plt.subplot(1, 3, 3), plt.imshow(image_contours)
    plt.title("Contours des feuilles")
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return {'nombre_feuilles_vertes': nombre_feuilles_vertes}

def process_nbre_feuille(filepath, output_path):
    # Charger et convertir l'image
    image = cv2.imread(filepath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Définir les plages HSV
    ranges = [
        {'lower': np.array([35, 40, 40]), 'upper': np.array([85, 255, 255])},  # Vert
        {'lower': np.array([20, 40, 40]), 'upper': np.array([35, 255, 255])},  # Jaune
        {'lower': np.array([10, 100, 40]), 'upper': np.array([20, 255, 255])},  # Orange
        {'lower': np.array([0, 100, 40]), 'upper': np.array([10, 255, 255])},  # Rouge bas
        {'lower': np.array([170, 100, 40]), 'upper': np.array([180, 255, 255])}  # Rouge haut
    ]

    # Initialiser le masque global
    masque_global = np.zeros_like(hsv[:, :, 0])

    # Créer chaque masque et le combiner
    for r in ranges:
        masque = cv2.inRange(hsv, r['lower'], r['upper'])
        masque_global = cv2.bitwise_or(masque_global, masque)

    # Nettoyage du masque élargi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(masque_global, cv2.MORPH_OPEN, kernel, iterations=2)


    # Trouver les contours principaux
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_result = image_rgb.copy()
    nombre_feuilles = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue  # Ignore les petits bruits

        # Trouver l'enveloppe convexe
        hull = cv2.convexHull(cnt, returnPoints=False)

        if hull is None or len(hull) < 3:
            continue

        # Trouver les défauts de convexité
        defects = cv2.convexityDefects(cnt, hull)

        feuille_count = 1  # On commence par 1 (la feuille elle-même)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, depth = defects[i, 0]
                if depth > 1000:  # Filtrer les petits creux (ajuster si besoin)
                    feuille_count += 1
                    # Dessiner le point du creux
                    far = tuple(cnt[f][0])
                    cv2.circle(image_result, far, 5, (0, 0, 255), -1)

        nombre_feuilles += feuille_count

        # Dessiner le contour global et l'enveloppe convexe
        cv2.drawContours(image_result, [cnt], -1, (0, 255, 0), 2)
        hull_points = cv2.convexHull(cnt)
        cv2.polylines(image_result, [hull_points], True, (255, 0, 0), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_result)
    plt.title(f"Nombre de feuilles estimé : {nombre_feuilles}", fontsize=12, color='blue')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return {'nombre_feuilles': nombre_feuilles}

def process_couleur_jaune(filepath, output_path):
    # Charger et convertir l'image
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Conversion en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Détection des parties jaunes
    lower_yellow = np.array([20, 100, 100])
    higher_yellow = np.array([30, 255, 255])
    masque_jaune = cv2.inRange(hsv, lower_yellow, higher_yellow)
    partie_jaune = cv2.bitwise_and(image, image, mask=masque_jaune)

    # Détection des contours des zones jaunes
    contours_jaunes, _ = cv2.findContours(masque_jaune, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nombre_zones_jaunes = len(contours_jaunes)
    print(f"Nombre de zones jaunes détectées : {nombre_zones_jaunes}")

    # Dessiner les contours jaunes sur l'image originale
    image_contours = cv2.drawContours(image.copy(), contours_jaunes, -1, (255, 255, 0), 2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(partie_jaune)
    plt.title("Parties jaunes détectées")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_contours)
    plt.title("Contours des zones jaunes")
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return {'nombre_zones_jaunes': nombre_zones_jaunes}

def process_croissance(dossier_images='uploads', dossier_resultats='results', nom_resultat='comparaison_tailles.png'):
    # Liste les fichiers image dans le dossier
    fichiers = sorted(
        [f for f in os.listdir(dossier_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        reverse=True
    )
    
    # Vérifie qu’il y a au moins 3 images
    if len(fichiers) < 3:
        raise ValueError("Pas assez d'images pour comparer (minimum 3 requises).")

    tailles = []
    images = []

    for i in range(3):
        chemin = os.path.join(dossier_images, fichiers[i])
        img = cv2.imread(chemin)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lower = np.array([60, 60, 60])
        higher = np.array([250, 250, 250])
        masque = cv2.inRange(img_rgb, lower, higher)
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError(f"Aucun contour détecté dans l'image : {fichiers[i]}")

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 5)

        tailles.append(h)
        images.append((img_rgb, h, fichiers[i]))

    # Affichage matplotlib
    plt.figure("Comparaison des tailles", figsize=(20, 4))
    for i, (img, h, nom) in enumerate(images):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.title(f'{nom}\nTaille: {h}', fontsize=10, color='blue')
        plt.axis('off')

    output_path = os.path.join(dossier_resultats, nom_resultat)
    plt.savefig(output_path)
    plt.close()

    return {
        "image_comparee": nom_resultat,
        "tailles": {
            fichiers[0]: tailles[0],
            fichiers[1]: tailles[1],
            fichiers[2]: tailles[2]
        }
    }

def process_diametre(filepath, output_path):
    image = cv2.imread(filepath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Détection des zones vertes
    lower_green = np.array([25, 40, 20])   # à ajuster selon ton éclairage
    upper_green = np.array([85, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Chercher les contours sur la partie verte uniquement
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Aucune zone verte détectée.")

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Dessiner un rectangle autour des parties vertes détectées
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Sauvegarder la figure
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f"Diamètre (largeur): {w} pixels", fontsize=12, color='blue')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

    return {"diametre_pixels": w}

def process_texture(filepath, output_path):

    image = cv2.imread(filepath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les plages de couleurs (vert, jaune, orange, rouge bas, rouge haut)
    color_ranges = [
        {'lower': np.array([25, 40, 20]), 'upper': np.array([85, 255, 255])},   # Vert
        {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])}, # Jaune
        {'lower': np.array([10, 100, 40]), 'upper': np.array([20, 255, 255])},  # Orange
        {'lower': np.array([0, 100, 40]),  'upper': np.array([10, 255, 255])},  # Rouge bas
        {'lower': np.array([170, 100, 40]), 'upper': np.array([180, 255, 255])} # Rouge haut
    ]

    # Créer le masque global combiné
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color in color_ranges:
        mask |= cv2.inRange(hsv, color['lower'], color['upper'])

    # Appliquer la détection de texture uniquement sur les zones sélectionnées
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Calcul de la variance du Laplacien
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    variance = laplacian.var()

    # Affichage
    plt.figure(figsize=(8, 6))
    plt.imshow(masked_gray, cmap='gray')
    plt.title(f"Indice de texture (variance Laplacien) : {variance:.2f}", fontsize=12, color='blue')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

    return {"texture_variance": variance}
