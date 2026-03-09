from flask import Flask, request, send_from_directory, jsonify
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from queue import Queue
from pathlib import Path
import algo
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Import pour la surveillance de fichiers
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    print("⚠️ watchdog non installé. Installez avec: pip install watchdog")
    WATCHDOG_AVAILABLE = False

app = Flask(__name__)

# Configuration des dossiers
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
WATCH_FOLDER = 'watch_folder'      # Nouveau : dossier surveillé
PROCESSED_FOLDER = 'processed'     # Nouveau : images traitées
ERROR_FOLDER = 'error_images'      # Nouveau : images en erreur

# Création des dossiers
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, WATCH_FOLDER, PROCESSED_FOLDER, ERROR_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Liste centralisée des algorithmes disponibles
ALGOS_DISPONIBLES = [
    'taille', 'couleur_verte', 'nbre_feuille',
    'couleur_jaune', 'croissance', 'diametre', 'texture'
]

# Configuration InfluxDB
INFLUX_CONFIG = {
    "url": "http://localhost:8086",
    "token": "aXcV_PcVBRNMq_1lRPeZYR5Cz5SqEGvndrbtuJqYIAEcTxWwSlTP4FHvfpZ98wq5A7z9HjRHep-haqVze9xsoQ==",
    "org": "Planty",
    "bucket": "test_plant-data"
}

# Connexion au client InfluxDB
client = InfluxDBClient(
    url=INFLUX_CONFIG["url"],
    token=INFLUX_CONFIG["token"],
    org=INFLUX_CONFIG["org"]
)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Queue pour gérer les images à traiter
processing_queue = Queue()

# Extensions d'images supportées
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Variable globale pour contrôler le mode surveillance
WATCH_MODE_ENABLED = True


def extract_plant_id(filename):
    """
    Extrait l'identifiant de la plante depuis le nom du fichier.
    Convention attendue : camX_planteY_*.jpg  (ex: cam1_plante2_photo.jpg)
    Retourne ex: "cam1_plante2". Si le format n'est pas respecté, retourne "inconnu".
    """
    import re
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Cherche un pattern du type cam\d+_plante\d+ n'importe où dans le nom
    match = re.search(r'(cam\d+_plante\d+)', basename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback : essaie juste camX
    match = re.search(r'(cam\d+)', basename, re.IGNORECASE)
    if match:
        print(f"⚠️ Pas de plante_id trouvé dans '{basename}', utilisation de '{match.group(1)}' seul.")
        return match.group(1).lower()
    print(f"⚠️ Impossible d'extraire un plant_id depuis '{basename}'. Utilisez le format: camX_planteY_nom.jpg")
    return "inconnu"

def create_influx_point(plant_id="inconnu"):
    """Crée un point InfluxDB pour une plante."""
    return Point("plante").tag("plante_id", plant_id).time(datetime.now(timezone.utc))

def process_single_algorithm(algo_name, filepath, result_path):
    """Traite un algorithme spécifique et retourne le résultat."""
    if algo_name == 'taille':
        return algo.process_taille(filepath, result_path)
    elif algo_name == 'couleur_verte':
        return algo.process_couleur_verte(filepath, result_path)
    elif algo_name == 'nbre_feuille':
        return algo.process_nbre_feuille(filepath, result_path)
    elif algo_name == 'couleur_jaune':
        return algo.process_couleur_jaune(filepath, result_path)
    elif algo_name == 'croissance':
        return algo.process_croissance(UPLOAD_FOLDER, RESULT_FOLDER)
    elif algo_name == 'diametre':
        return algo.process_diametre(filepath, result_path)
    elif algo_name == 'texture':
        return algo.process_texture(filepath, result_path)
    else:
        raise ValueError(f"Algorithme non supporté: {algo_name}")

def add_result_to_influx_point(point, algo_name, result_data):
    """Ajoute les résultats d'un algorithme au point InfluxDB."""
    field_mappings = {
        'taille': lambda data: {'taille': data.get('taille')},
        'couleur_verte': lambda data: {'couleur_verte': data.get('nombre_feuilles_vertes')},
        'nbre_feuille': lambda data: {'nombre_feuilles': data.get('nombre_feuilles')},
        'couleur_jaune': lambda data: {'couleur_jaune': data.get('nombre_zones_jaunes')},
        'croissance': lambda data: {'croissance': data if not isinstance(data, dict) else data.get('croissance')},
        'diametre': lambda data: {'diametre': data.get('diametre_pixels')},
        'texture': lambda data: {'texture_variance': data.get('texture_variance')}
    }
    
    if algo_name in field_mappings:
        fields = field_mappings[algo_name](result_data)
        for field_name, field_value in fields.items():
            if field_value is not None:
                point = point.field(field_name, field_value)
    
    return point

def is_image_file(filepath):
    """Vérifie si le fichier est une image supportée."""
    return Path(filepath).suffix.lower() in SUPPORTED_EXTENSIONS

def is_file_complete(filepath, wait_time=1):
    """Vérifie si le fichier est complètement écrit (pas en cours de copie)."""
    try:
        # Vérifier la taille du fichier deux fois avec un délai
        size1 = os.path.getsize(filepath)
        time.sleep(wait_time)
        size2 = os.path.getsize(filepath)
        return size1 == size2 and size1 > 0
    except (OSError, FileNotFoundError):
        return False

def process_image_automatically(image_path):
    """Traite automatiquement une image avec tous les algorithmes disponibles."""
    print(f"🔄 Début du traitement automatique: {os.path.basename(image_path)}")
    
    # Timestamp pour les fichiers de résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(image_path).stem
    
    # Extraction de l'identifiant plante depuis le nom du fichier
    plant_id = extract_plant_id(image_path)
    print(f"  🌿 Plante identifiée : {plant_id}")

    # Statistiques de traitement
    results = {}
    errors = {}
    point = create_influx_point(plant_id)
    
    # Traitement de tous les algorithmes
    for algo_name in ALGOS_DISPONIBLES:
        try:
            print(f"  🧮 Traitement {algo_name}...")
            
            # Génération du nom de fichier résultat
            result_filename = f"{algo_name}_{timestamp}_{original_name}.png"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            
            # Traitement de l'algorithme
            result_data = process_single_algorithm(algo_name, image_path, result_path)
            
            # Stockage du résultat
            results[algo_name] = {
                "data": result_data,
                "result_path": result_path if algo_name != 'croissance' else None,
                "status": "success",
                "timestamp": timestamp
            }
            
            # Ajout au point InfluxDB
            point = add_result_to_influx_point(point, algo_name, result_data)
            
            print(f"    ✅ {algo_name} terminé")
            
        except Exception as e:
            error_msg = str(e)
            errors[algo_name] = {
                "error": error_msg,
                "status": "failed",
                "timestamp": timestamp
            }
            print(f"    ❌ {algo_name} échoué: {error_msg}")
    
    # Sauvegarde en InfluxDB si au moins un algorithme a réussi
    if results:
        try:
            # Ajout du nom de fichier comme tag
            point = point.tag("image_file", os.path.basename(image_path))
            point = point.tag("processing_mode", "automatic")
            
            write_api.write(
                bucket=INFLUX_CONFIG["bucket"], 
                org=INFLUX_CONFIG["org"], 
                record=point
            )
            print(f"  💾 Données sauvegardées dans InfluxDB")
        except Exception as e:
            print(f"  ⚠️ Erreur InfluxDB: {str(e)}")
    
    # Déplacement de l'image traitée
    try:
        processed_path = os.path.join(PROCESSED_FOLDER, f"{timestamp}_{os.path.basename(image_path)}")
        shutil.move(image_path, processed_path)
        print(f"  📁 Image déplacée vers: {processed_path}")
    except Exception as e:
        print(f"  ⚠️ Erreur lors du déplacement: {str(e)}")
        try:
            # En cas d'erreur, déplacer vers le dossier d'erreur
            error_path = os.path.join(ERROR_FOLDER, f"error_{timestamp}_{os.path.basename(image_path)}")
            shutil.move(image_path, error_path)
            print(f"  📁 Image déplacée vers dossier d'erreur: {error_path}")
        except Exception as e2:
            print(f"  ❌ Impossible de déplacer l'image: {str(e2)}")
    
    # Résumé final
    success_count = len(results)
    error_count = len(errors)
    print(f"✅ Traitement terminé: {success_count} succès, {error_count} erreurs")
    
    return {
        "image_path": image_path,
        "timestamp": timestamp,
        "results": results,
        "errors": errors,
        "summary": {
            "success_count": success_count,
            "error_count": error_count,
            "total_algorithms": len(ALGOS_DISPONIBLES)
        }
    }

def worker_thread():
    """Thread worker qui traite les images de la queue."""
    print("🔧 Worker thread démarré")
    
    while True:
        try:
            # Récupération d'une image à traiter (bloquant)
            image_path = processing_queue.get()
            
            if image_path is None:  # Signal d'arrêt
                break
            
            # Vérification que le fichier existe encore
            if os.path.exists(image_path):
                # Attendre que le fichier soit complètement écrit
                if is_file_complete(image_path):
                    process_image_automatically(image_path)
                else:
                    print(f"⚠️ Fichier incomplet ignoré: {image_path}")
            else:
                print(f"⚠️ Fichier disparu: {image_path}")
            
            # Marquer la tâche comme terminée
            processing_queue.task_done()
            
        except Exception as e:
            print(f"❌ Erreur dans le worker thread: {str(e)}")
            processing_queue.task_done()

class ImageWatchHandler(FileSystemEventHandler):
    """Handler pour surveiller les nouveaux fichiers images."""
    
    def on_created(self, event):
        """Appelé quand un nouveau fichier est créé."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Vérifier si c'est une image
        if is_image_file(file_path):
            print(f"📸 Nouvelle image détectée: {os.path.basename(file_path)}")
            
            # Attendre un peu pour s'assurer que le fichier est complètement écrit
            time.sleep(0.5)
            
            # Ajouter à la queue de traitement
            processing_queue.put(file_path)
        else:
            print(f"⚠️ Fichier non-image ignoré: {os.path.basename(file_path)}")
    
    def on_moved(self, event):
        """Appelé quand un fichier est déplacé (peut être un nouveau fichier)."""
        if event.is_directory:
            return
        
        # Traiter comme un nouveau fichier
        self.on_created(event)

def start_folder_watcher():
    """Démarre la surveillance du dossier."""
    if not WATCHDOG_AVAILABLE:
        print("❌ Surveillance de dossier désactivée (watchdog non installé)")
        return None
    
    print(f"👁️ Démarrage de la surveillance du dossier: {WATCH_FOLDER}")
    
    event_handler = ImageWatchHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()
    
    return observer

def process_existing_images():
    """Traite les images déjà présentes dans le dossier surveillé au démarrage."""
    existing_images = []
    
    for filename in os.listdir(WATCH_FOLDER):
        filepath = os.path.join(WATCH_FOLDER, filename)
        if os.path.isfile(filepath) and is_image_file(filepath):
            existing_images.append(filepath)
    
    if existing_images:
        print(f"📁 {len(existing_images)} image(s) existante(s) trouvée(s)")
        for image_path in existing_images:
            print(f"  📸 Ajout à la queue: {os.path.basename(image_path)}")
            processing_queue.put(image_path)
    else:
        print("📁 Aucune image existante dans le dossier surveillé")

# =============================================
# ENDPOINTS API (inchangés pour compatibilité)
# =============================================

@app.route('/algos', methods=['GET'])
def get_available_algorithms():
    """Endpoint pour récupérer la liste des algorithmes disponibles."""
    return jsonify({
        "algorithms": ALGOS_DISPONIBLES,
        "count": len(ALGOS_DISPONIBLES)
    })

# Modifications à apporter dans votre app.py

# 1. Modifier l'endpoint /upload pour sauver dans watch_folder
@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint pour uploader une image vers le dossier surveillé."""
    
    # Validation de l'image
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "Nom de fichier invalide"}), 400

    # Sauvegarde de l'image dans watch_folder en conservant le nom d'origine
    # IMPORTANT : le nom doit respecter la convention camX_planteY_*.ext
    # On ajoute le timestamp en suffixe pour éviter les doublons sans écraser l'info caméra/plante
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name, ext = os.path.splitext(image.filename)
    filename = f"{original_name}_{timestamp}{ext}"
    
    # CHANGEMENT PRINCIPAL : Sauver dans watch_folder au lieu de uploads
    filepath = os.path.join(WATCH_FOLDER, filename)
    
    try:
        image.save(filepath)
        
        # L'image sera automatiquement détectée et traitée par le système de surveillance
        return jsonify({
            "message": "Image uploadée avec succès",
            "filename": filename,
            "timestamp": timestamp,
            "status": "En cours de traitement automatique",
            "watch_folder": WATCH_FOLDER
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la sauvegarde: {str(e)}"}), 500

# 2. Nouvel endpoint pour l'upload avec traitement immédiat (optionnel)
@app.route('/upload_immediate', methods=['POST'])
def upload_image_immediate():
    """Endpoint pour uploader et traiter immédiatement (ancien comportement)."""
    
    # Validation de l'image
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "Nom de fichier invalide"}), 400

    # Récupération et validation des algorithmes
    algos_raw = request.form.get('algo', '')
    if not algos_raw:
        return jsonify({"error": "Aucun algorithme spécifié"}), 400

    # Support des algorithmes multiples séparés par virgules
    algo_list = []
    for algo in algos_raw.split(','):
        algo = algo.strip()
        if algo in ALGOS_DISPONIBLES:
            algo_list.append(algo)
        elif algo:
            return jsonify({
                "error": f"Algorithme '{algo}' non supporté",
                "available_algorithms": ALGOS_DISPONIBLES
            }), 400

    if not algo_list:
        return jsonify({
            "error": "Aucun algorithme valide fourni",
            "available_algorithms": ALGOS_DISPONIBLES
        }), 400

    # Sauvegarde temporaire pour traitement immédiat
    # IMPORTANT : le nom doit respecter la convention camX_planteY_*.ext
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name, ext = os.path.splitext(image.filename)
    filename = f"{original_name}_{timestamp}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        image.save(filepath)
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la sauvegarde: {str(e)}"}), 500

    # Traitement des algorithmes (code existant)
    results = {}
    errors = {}
    plant_id = extract_plant_id(filename)
    print(f"  🌿 Plante identifiée : {plant_id}")
    point = create_influx_point(plant_id)
    
    for algo_name in algo_list:
        try:
            result_filename = f"{algo_name}_{timestamp}_{original_name}.png"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            
            result_data = process_single_algorithm(algo_name, filepath, result_path)
            
            results[algo_name] = {
                "data": result_data,
                "result_url": f"/result/{result_filename}" if algo_name != 'croissance' else None,
                "status": "success"
            }
            
            point = add_result_to_influx_point(point, algo_name, result_data)
            
        except Exception as e:
            errors[algo_name] = {
                "error": str(e),
                "status": "failed"
            }

    # Sauvegarde en InfluxDB
    if results:
        try:
            point = point.tag("processing_mode", "api_immediate")
            point = point.tag("image_file", filename)
            
            write_api.write(
                bucket=INFLUX_CONFIG["bucket"], 
                org=INFLUX_CONFIG["org"], 
                record=point
            )
        except Exception as e:
            print(f"Erreur InfluxDB: {str(e)}")

    # Construction de la réponse
    response_data = {
        "message": f"Traitement immédiat terminé pour {len(results)} algorithme(s)",
        "uploaded_file": filename,
        "timestamp": timestamp,
        "results": results
    }
    
    if errors:
        response_data["errors"] = errors
        response_data["message"] += f" avec {len(errors)} erreur(s)"

    # Statut HTTP selon le résultat
    if results and not errors:
        return jsonify(response_data), 200
    elif results and errors:
        return jsonify(response_data), 207
    else:
        return jsonify(response_data), 500

# 3. Endpoint pour vérifier le statut d'une image en traitement
@app.route('/status/<filename>', methods=['GET'])
def check_processing_status(filename):
    """Vérifie le statut de traitement d'une image."""
    
    # Vérifier si l'image est dans watch_folder (en attente)
    watch_path = os.path.join(WATCH_FOLDER, filename)
    if os.path.exists(watch_path):
        return jsonify({
            "filename": filename,
            "status": "pending",
            "message": "Image en attente de traitement",
            "queue_size": processing_queue.qsize()
        })
    
    # Vérifier si l'image a été traitée (dans processed)
    processed_files = [f for f in os.listdir(PROCESSED_FOLDER) if filename in f]
    if processed_files:
        return jsonify({
            "filename": filename,
            "status": "completed",
            "message": "Image traitée avec succès",
            "processed_files": processed_files
        })
    
    # Vérifier si l'image est en erreur
    error_files = [f for f in os.listdir(ERROR_FOLDER) if filename in f]
    if error_files:
        return jsonify({
            "filename": filename,
            "status": "error",
            "message": "Erreur lors du traitement",
            "error_files": error_files
        })
    
    # Image non trouvée
    return jsonify({
        "filename": filename,
        "status": "not_found",
        "message": "Image non trouvée dans le système"
    }), 404

@app.route('/result/<filename>', methods=['GET'])
def view_result(filename):
    """Endpoint pour récupérer les images résultats."""
    try:
        return send_from_directory(RESULT_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({"error": "Fichier non trouvé"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé pour vérifier que l'API fonctionne."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithms_available": len(ALGOS_DISPONIBLES),
        "watch_mode_enabled": WATCH_MODE_ENABLED,
        "watchdog_available": WATCHDOG_AVAILABLE,
        "queue_size": processing_queue.qsize()
    })

@app.route('/watch/status', methods=['GET'])
def watch_status():
    """Endpoint pour récupérer le statut de la surveillance."""
    return jsonify({
        "watch_folder": WATCH_FOLDER,
        "watch_mode_enabled": WATCH_MODE_ENABLED,
        "watchdog_available": WATCHDOG_AVAILABLE,
        "queue_size": processing_queue.qsize(),
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "folders": {
            "watch": WATCH_FOLDER,
            "processed": PROCESSED_FOLDER,
            "results": RESULT_FOLDER,
            "errors": ERROR_FOLDER
        }
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Gestion des fichiers trop volumineux."""
    return jsonify({"error": "Fichier trop volumineux"}), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Gestion des erreurs internes."""
    return jsonify({"error": "Erreur interne du serveur"}), 500

# ========================================
# DÉMARRAGE DE L'APPLICATION
# ========================================

if __name__ == '__main__':
    print("🌱 Démarrage du serveur d'analyse de plantes")
    print("=" * 50)
    print(f"📁 Dossier surveillé: {WATCH_FOLDER}")
    print(f"📁 Dossier uploads: {UPLOAD_FOLDER}")
    print(f"📁 Dossier résultats: {RESULT_FOLDER}")
    print(f"📁 Dossier images traitées: {PROCESSED_FOLDER}")
    print(f"📁 Dossier images en erreur: {ERROR_FOLDER}")
    print(f"🔗 InfluxDB: {INFLUX_CONFIG['url']}")
    print(f"🚀 {len(ALGOS_DISPONIBLES)} algorithmes disponibles")
    
    # Démarrage du thread worker
    worker = threading.Thread(target=worker_thread, daemon=True)
    worker.start()
    
    # Traitement des images existantes
    process_existing_images()
    
    # Démarrage de la surveillance de dossier
    observer = start_folder_watcher()
    
    if observer:
        print("👁️ Surveillance de dossier activée")
        print(f"💡 Ajoutez des images dans '{WATCH_FOLDER}' pour un traitement automatique")
    else:
        print("⚠️ Surveillance de dossier désactivée")
    
    print("=" * 50)
    print("🌐 Serveur Flask démarré sur http://0.0.0.0:5000")
    print("📖 Endpoints disponibles:")
    print("   GET  /health - Statut du serveur")
    print("   GET  /algos - Liste des algorithmes")
    print("   GET  /watch/status - Statut de la surveillance")
    print("   POST /upload - Upload manuel (compatibilité)")
    print("   GET  /result/<filename> - Récupération des résultats")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du serveur...")
        if observer:
            observer.stop()
            observer.join()
        
        # Arrêt du worker thread
        processing_queue.put(None)
        worker.join()
        
        print("👋 Serveur arrêté proprement")