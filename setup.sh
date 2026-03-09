#!/bin/bash
# ============================================================
# setup.sh — Mise en place de l'environnement Planty
# Usage : bash setup.sh
# ============================================================

set -e  # Arrêt immédiat en cas d'erreur

VENV_DIR=".venv"
PYTHON_MIN="3.10"

echo "============================================================"
echo "  🌱 Installation de l'environnement Planty"
echo "============================================================"

# ── 1. Vérification de Python ────────────────────────────────
echo ""
echo "🔍 Vérification de Python..."

if ! command -v python3 &>/dev/null; then
    echo "❌ python3 introuvable. Installez Python $PYTHON_MIN ou supérieur."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "   Version détectée : Python $PYTHON_VERSION"

# Vérification version minimale
python3 -c "
import sys
if sys.version_info < (3, 10):
    print('❌ Python 3.10 minimum requis.')
    sys.exit(1)
"

# ── 2. Création du virtualenv ────────────────────────────────
echo ""
echo "📦 Création du virtualenv dans '$VENV_DIR'..."

if [ -d "$VENV_DIR" ]; then
    echo "   ⚠️  Virtualenv déjà existant, suppression et recréation..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
echo "   ✅ Virtualenv créé"

# ── 3. Activation + mise à jour pip ─────────────────────────
echo ""
echo "⚙️  Activation et mise à jour de pip..."

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
echo "   ✅ pip à jour : $(pip --version)"

# ── 4. Installation des dépendances ─────────────────────────
echo ""
echo "📥 Installation des dépendances (requirements.txt)..."

pip install -r requirements.txt

echo "   ✅ Dépendances installées"

# ── 5. Création des dossiers de travail ──────────────────────
echo ""
echo "📁 Création des dossiers de travail..."

for dir in uploads results watch_folder processed error_images; do
    mkdir -p "$dir"
    echo "   ✅ $dir/"
done

# ── 6. Résumé ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅ Installation terminée avec succès !"
echo "============================================================"
echo ""
echo "  Pour démarrer le serveur :"
echo ""
echo "    source $VENV_DIR/bin/activate"
echo "    python app.py"
echo ""
echo "  Pour quitter le virtualenv :"
echo "    deactivate"
echo "============================================================"
