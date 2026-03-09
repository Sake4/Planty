import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math

# ============================================================
#  MODULE COMMUN — Prétraitement partagé
# ============================================================

# Plages HSV de référence pour la végétation
COLOR_RANGES = [
    {'lower': np.array([25, 40, 20]),   'upper': np.array([85, 255, 255])},   # Vert
    {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},   # Jaune
    {'lower': np.array([10, 100, 40]),  'upper': np.array([20, 255, 255])},   # Orange
    {'lower': np.array([0,  100, 40]),  'upper': np.array([10, 255, 255])},   # Rouge bas
    {'lower': np.array([170, 100, 40]), 'upper': np.array([180, 255, 255])},  # Rouge haut
]

GREEN_RANGE = {
    'lower': np.array([25, 40, 20]),
    'upper': np.array([85, 255, 255]),
}

YELLOW_RANGE = {
    'lower': np.array([20, 100, 100]),
    'upper': np.array([30, 255, 255]),
}


def _charger_image(filepath):
    """Charge une image et la retourne en RGB + HSV."""
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {filepath}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image_rgb, hsv


def _nettoyer_masque(masque, kernel_size=5, iterations=2):
    """
    Nettoyage morphologique :
      - Ouverture  → supprime les petits artéfacts isolés (insectes, poussières)
      - Fermeture  → bouche les lacunes internes aux feuilles
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ouvert  = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  kernel, iterations=iterations)
    ferme   = cv2.morphologyEx(ouvert, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return ferme


def _segmenter_vegetation(hsv, color_ranges=None):
    """
    Crée un masque binaire combinant toutes les plages de couleur végétation.
    Applique ensuite un nettoyage morphologique.
    """
    if color_ranges is None:
        color_ranges = COLOR_RANGES
    masque = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in color_ranges:
        masque |= cv2.inRange(hsv, r['lower'], r['upper'])
    return _nettoyer_masque(masque)


def _segmenter_taille(hsv):
    """
    Segmentation robuste pour la hauteur de la plante :
    utilise le canal V (luminosité) + seuillage d'Otsu pour ne pas
    dépendre de couleurs absolues.
    """
    # Extraire la couche V et appliquer Otsu
    v_channel = hsv[:, :, 2]
    _, masque_otsu = cv2.threshold(v_channel, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Combiner avec la segmentation végétation pour limiter aux zones plante
    masque_veg = _segmenter_vegetation(hsv)
    masque = cv2.bitwise_and(masque_otsu, masque_veg)
    return _nettoyer_masque(masque)


def _sauvegarder_figure(output_path):
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


# ============================================================
#  1. TAILLE (hauteur en pixels)
# ============================================================

def process_taille(filepath, output_path):
    """
    Mesure la hauteur de la plante via bounding rect du plus grand contour.
    Segmentation HSV + Otsu (remplace l'ancien masque RGB fragile).
    """
    image_rgb, hsv = _charger_image(filepath)
    masque = _segmenter_taille(hsv)

    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    taille = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        x1, y1, w1, h1 = cv2.boundingRect(c)
        cv2.rectangle(image_rgb, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 5)
        taille = h1
        title = f'Taille : {h1} px'
    else:
        title = 'Aucun contour trouvé'

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(title, fontsize=12, color='blue')
    plt.axis('off')
    _sauvegarder_figure(output_path)

    return {'taille': taille}


# ============================================================
#  2. COULEUR VERTE (biomasse saine)
# ============================================================

def process_couleur_verte(filepath, output_path):
    """
    Détecte et quantifie les zones vertes (biomasse saine).
    Nettoyage morphologique appliqué avant le comptage des contours.
    """
    image_rgb, hsv = _charger_image(filepath)

    masque_verte = cv2.inRange(hsv, GREEN_RANGE['lower'], GREEN_RANGE['upper'])
    masque_verte = _nettoyer_masque(masque_verte)

    partie_verte = cv2.bitwise_and(image_rgb, image_rgb, mask=masque_verte)

    contours_feuilles, _ = cv2.findContours(masque_verte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Filtrer les micro-contours (bruit résiduel)
    contours_feuilles = [c for c in contours_feuilles if cv2.contourArea(c) > 300]
    nombre_feuilles_vertes = len(contours_feuilles)

    image_contours = image_rgb.copy()
    cv2.drawContours(image_contours, contours_feuilles, -1, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    axes[0].imshow(masque_verte, cmap='gray')
    axes[0].set_title("Masque vert")
    axes[0].axis('off')
    axes[1].imshow(partie_verte)
    axes[1].set_title("Parties vertes détectées")
    axes[1].axis('off')
    axes[2].imshow(image_contours)
    axes[2].set_title(f"Contours feuilles ({nombre_feuilles_vertes})")
    axes[2].axis('off')
    plt.tight_layout()
    _sauvegarder_figure(output_path)

    return {'nombre_feuilles_vertes': nombre_feuilles_vertes}


# ============================================================
#  3. NOMBRE DE FEUILLES (convexité + règle du cosinus)
# ============================================================

def _angle_defaut(cnt, s, e, f):
    """
    Calcule l'angle au sommet du défaut de convexité via la règle du cosinus.
    Retourne l'angle en degrés.
    """
    start = tuple(cnt[s][0])
    end   = tuple(cnt[e][0])
    far   = tuple(cnt[f][0])

    a = math.dist(end,   start)  # côté opposé au sommet far
    b = math.dist(far,   start)
    c = math.dist(end,   far)

    # Éviter division par zéro
    denom = 2 * b * c
    if denom == 0:
        return 180.0

    cos_angle = (b**2 + c**2 - a**2) / denom
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp numérique
    return math.degrees(math.acos(cos_angle))


def process_nbre_feuille(filepath, output_path):
    """
    Compte les feuilles par analyse de convexité.
    Améliorations :
      - Filtre cosinus (angle < 90°) pour valider les séparations réelles
      - Seuil de profondeur en pixels réels (depth/256 > 8 px)
      - Nettoyage morphologique Ouverture + Fermeture
    """
    image_rgb, hsv = _charger_image(filepath)
    masque = _segmenter_vegetation(hsv)

    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_result = image_rgb.copy()
    nombre_feuilles = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue

        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) < 3:
            continue

        defects = cv2.convexityDefects(cnt, hull)
        feuille_count = 1  # la plante elle-même compte pour 1

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, depth = defects[i, 0]
                depth_px = depth / 256.0  # OpenCV stocke depth * 256

                # Critère 1 : profondeur minimale de 8 px (filtre le bruit de contour)
                if depth_px < 8:
                    continue

                # Critère 2 : angle aigu (<= 90°) → vraie séparation entre feuilles
                angle = _angle_defaut(cnt, s, e, f)
                if angle <= 90:
                    feuille_count += 1
                    far = tuple(cnt[f][0])
                    cv2.circle(image_result, far, 6, (0, 0, 255), -1)

        nombre_feuilles += feuille_count

        cv2.drawContours(image_result, [cnt], -1, (0, 255, 0), 2)
        hull_pts = cv2.convexHull(cnt)
        cv2.polylines(image_result, [hull_pts], True, (255, 0, 0), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_result)
    plt.title(f"Nombre de feuilles estimé : {nombre_feuilles}", fontsize=12, color='blue')
    plt.axis('off')
    _sauvegarder_figure(output_path)

    return {'nombre_feuilles': nombre_feuilles}


# ============================================================
#  4. COULEUR JAUNE (stress nutritionnel / hydrique)
# ============================================================

def process_couleur_jaune(filepath, output_path):
    """
    Détecte les zones jaunes (indicateur de stress).
    Nettoyage morphologique + filtrage des micro-contours.
    """
    image_rgb, hsv = _charger_image(filepath)

    masque_jaune = cv2.inRange(hsv, YELLOW_RANGE['lower'], YELLOW_RANGE['upper'])
    masque_jaune = _nettoyer_masque(masque_jaune)

    partie_jaune = cv2.bitwise_and(image_rgb, image_rgb, mask=masque_jaune)

    contours_jaunes, _ = cv2.findContours(masque_jaune, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_jaunes = [c for c in contours_jaunes if cv2.contourArea(c) > 200]
    nombre_zones_jaunes = len(contours_jaunes)

    image_contours = image_rgb.copy()
    cv2.drawContours(image_contours, contours_jaunes, -1, (255, 215, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_rgb);      axes[0].set_title("Image originale");         axes[0].axis('off')
    axes[1].imshow(partie_jaune);   axes[1].set_title("Zones jaunes détectées");  axes[1].axis('off')
    axes[2].imshow(image_contours); axes[2].set_title(f"Contours ({nombre_zones_jaunes})"); axes[2].axis('off')
    plt.tight_layout()
    _sauvegarder_figure(output_path)

    return {'nombre_zones_jaunes': nombre_zones_jaunes}


# ============================================================
#  5. CROISSANCE (comparaison temporelle de 3 images)
# ============================================================

def process_croissance(dossier_images='uploads', dossier_resultats='results',
                       nom_resultat='comparaison_tailles.png'):
    """
    Compare la hauteur de la plante sur 3 images successives.
    Utilise désormais la segmentation HSV + Otsu (cohérente avec process_taille).
    Les fichiers sont triés par nom (convention : horodatage dans le nom de fichier).
    """
    fichiers = sorted(
        [f for f in os.listdir(dossier_images)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    if len(fichiers) < 3:
        raise ValueError("Minimum 3 images requises pour la comparaison de croissance.")

    # Prendre les 3 plus récentes (fin de liste = plus récent si trié chronologiquement)
    fichiers_selectionnes = fichiers[-3:]
    tailles = []
    images  = []

    for nom in fichiers_selectionnes:
        chemin = os.path.join(dossier_images, nom)
        image_rgb, hsv = _charger_image(chemin)
        masque = _segmenter_taille(hsv)

        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError(f"Aucun contour détecté dans : {nom}")

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 5)

        tailles.append(h)
        images.append((image_rgb, h, nom))

    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    fig.suptitle("Comparaison de croissance", fontsize=14, color='darkgreen')
    for i, (img, h, nom) in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f'{nom}\nHauteur : {h} px', fontsize=9, color='blue')
        axes[i].axis('off')
    plt.tight_layout()

    output_path = os.path.join(dossier_resultats, nom_resultat)
    os.makedirs(dossier_resultats, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    return {
        "image_comparee": nom_resultat,
        "tailles": {fichiers_selectionnes[i]: tailles[i] for i in range(3)}
    }


# ============================================================
#  6. DIAMÈTRE (largeur de la tige / canopée)
# ============================================================

def process_diametre(filepath, output_path):
    """
    Mesure le diamètre (largeur) de la zone verte principale.
    Nettoyage morphologique pour éviter que des feuilles collées
    ne faussent la largeur de la tige.
    """
    image_rgb, hsv = _charger_image(filepath)

    masque = cv2.inRange(hsv, GREEN_RANGE['lower'], GREEN_RANGE['upper'])
    masque = _nettoyer_masque(masque, kernel_size=7, iterations=1)

    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Aucune zone verte détectée.")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 4)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f"Diamètre (largeur) : {w} px", fontsize=12, color='blue')
    plt.axis('off')
    _sauvegarder_figure(output_path)

    return {"diametre_pixels": w}


# ============================================================
#  7. TEXTURE (rugosité foliaire via variance du Laplacien)
# ============================================================

def process_texture(filepath, output_path):
    """
    Calcule l'indice de texture (variance du Laplacien) sur les zones végétation.
    Une variance élevée indique une surface rugueuse → détection précoce de maladies.
    """
    image_rgb, hsv = _charger_image(filepath)
    masque = _segmenter_vegetation(hsv)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=masque)

    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    variance  = float(laplacian.var())

    # Visualisation côte à côte : zone masquée + carte Laplacien
    laplacian_norm = cv2.normalize(
        np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(masked_gray, cmap='gray')
    axes[0].set_title("Zone végétation (niveaux de gris)")
    axes[0].axis('off')
    axes[1].imshow(laplacian_norm, cmap='hot')
    axes[1].set_title(f"Carte Laplacien\nVariance : {variance:.2f}")
    axes[1].axis('off')
    plt.suptitle("Indice de texture foliaire", fontsize=13, color='blue')
    plt.tight_layout()
    _sauvegarder_figure(output_path)

    return {"texture_variance": variance}