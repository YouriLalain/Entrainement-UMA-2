#!/bin/bash
# Script de test rapide sur RunPod

echo "🚀 TEST RAPIDE DU MODÈLE"
echo "========================"
echo ""

# Vérifie qu'une image est fournie
if [ -z "$1" ]; then
    echo "❌ Usage: ./quick_test.sh <image_path> [question]"
    echo ""
    echo "Exemples:"
    echo "  ./quick_test.sh dataset/images/airplane1.jpg"
    echo "  ./quick_test.sh test.jpg \"Quelles sont les parties de cet avion?\""
    exit 1
fi

IMAGE_PATH=$1
QUESTION=${2:-"Quelles parties de l'avion vois-tu ?"}

# Trouve le dernier checkpoint
LATEST_RUN=$(ls -td experiments/run_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ Aucun run trouvé dans experiments/"
    exit 1
fi

CHECKPOINT="$LATEST_RUN/checkpoints/checkpoint_best.pt"

echo "📁 Run: $LATEST_RUN"
echo "🎯 Checkpoint: $CHECKPOINT"
echo "🖼️  Image: $IMAGE_PATH"
echo "❓ Question: $QUESTION"
echo ""

# Lance le test
python test_interactive.py \
    --image "$IMAGE_PATH" \
    --question "$QUESTION" \
    --checkpoint "$CHECKPOINT" \
    --confidence 0.3 \
    --top-k 20

echo ""
echo "✅ Résultat sauvegardé dans: result_*.jpg"
