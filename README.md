# 🚀 SAIL-VL Detection Training System

Système d'entraînement pour ajouter une tête de détection au VLM SAIL-VL2-2B.

## 📋 Architecture

```
Image → Vision Encoder (FROZEN)
           ↓
     Detection Head (TRAINABLE) → Bounding Boxes
           ↓
     ROI Projector (TRAINABLE) → Object Tokens
           ↓
     LLM (FROZEN) → Texte + Détections
```

## 🛠️ Installation

```bash
# Installe les dépendances
pip install -r requirements.txt
```

## 📊 Dataset

Structure attendue :
```
dataset/
├── labels_my-project-name_2025-10-19-08-36-47.csv
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

CSV avec colonnes :
- `label_name`, `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`
- `image_name`, `image_width`, `image_height`

## 🚀 Utilisation

### 1. Préparer le dataset

```bash
python prepare_dataset.py
```

Génère :
- `dataset/processed/train.json`
- `dataset/processed/val.json`

### 2. Entraîner

```bash
python train.py
```

Crée automatiquement un nouveau run :
```
experiments/
└── run_001_2025-10-19_14-30/
    ├── checkpoints/
    ├── logs/
    ├── config.yaml
    └── results/
```

### 3. Inférence

```bash
python inference.py \
    --image dataset/images/test.jpg \
    --checkpoint experiments/run_001_2025-10-19_14-30/checkpoints/checkpoint_best.pt \
    --output result.jpg
```

## ⚙️ Configuration

Édite `config.yaml` pour changer :
- Nombre d'epochs
- Batch size
- Learning rate
- Device (mps/cpu)

## 📦 Fichiers

- **prepare_dataset.py** : Convertit le CSV en format d'entraînement
- **model.py** : Architecture du modèle
- **train.py** : Script d'entraînement avec gestion des runs
- **inference.py** : Test sur une image
- **config.yaml** : Configuration

## 🔧 Troubleshooting

### MPS ne fonctionne pas
Change dans `config.yaml` :
```yaml
device: "cpu"
```

### Out of memory
Réduis le batch_size dans `config.yaml` :
```yaml
batch_size: 1
```

### Import Error
```bash
pip install --upgrade transformers torch torchvision
```

## 📝 Notes

- Le VLM SAIL-VL est **complètement frozen**
- Seulement la Detection Head et le ROI Projector sont entraînés
- Chaque run crée un nouveau dossier automatiquement
- Compatible Mac Apple Silicon (MPS) et CPU
- Modèle plus léger que OVIS (2B vs 3B paramètres)

## 🎯 Classes détectées (12)

1. fuselage
2. cockpit
3. derive
4. empennage
5. aile
6. aileron
7. volet
8. reacteur
9. nacelle_moteur
10. tuyere
11. train_atterrissage
12. porte
