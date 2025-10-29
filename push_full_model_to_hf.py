#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour pusher le MODÈLE COMPLET (InternVL2 + tête) sur Hugging Face Hub
"""

import os
import argparse
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import yaml


def create_model_card(repo_name, config_path, output_path):
    """Crée un README.md pour le modèle"""
    
    # Charge la config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    classes = config.get('classes', [])
    num_classes = len(classes)
    
    readme_content = f"""---
license: mit
base_model: OpenGVLab/InternVL2-2B
tags:
  - vision
  - object-detection
  - aircraft
  - internvl
  - fine-tuned
language:
  - en
  - fr
pipeline_tag: object-detection
---

# InternVL2-2B Aircraft Parts Detection

Modèle InternVL2-2B fine-tuné pour la détection de {num_classes} parties d'avion.

## 🎯 Description

Ce modèle est basé sur **InternVL2-2B** (OpenGVLab) et a été fine-tuné avec une tête de détection personnalisée pour détecter les parties suivantes d'un avion :

{chr(10).join(f"- {i+1}. **{cls}**" for i, cls in enumerate(classes))}

## 🏗️ Architecture

```
Image (448x448)
    ↓
InternVL2-2B Vision Encoder (FROZEN)
    ↓
Detection Head (TRAINABLE)
    ├── Classification: {num_classes} classes
    └── Bounding Boxes: (x, y, w, h)
    ↓
ROI Projector (TRAINABLE)
    ↓
Object Tokens → LLM
```

**Paramètres entraînés** : ~2M (Detection Head + ROI Projector)  
**Paramètres frozen** : ~2B (InternVL2-2B)

## 🚀 Utilisation

### Installation

```bash
pip install torch torchvision transformers pillow pyyaml
```

### Chargement du modèle

```python
import torch
from model import load_model
import yaml

# Charge la config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Charge le modèle complet (InternVL2 + tête)
model = load_model(config)

# Charge les poids de la tête de détection
checkpoint = torch.load("detection_weights.pt")
model.detection_head.load_state_dict(checkpoint['detection_head'])
model.roi_projector.load_state_dict(checkpoint['roi_projector'])

model.eval()
```

### Inférence

```python
from PIL import Image
import torchvision.transforms as transforms

# Prépare l'image
image = Image.open("aircraft.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
image_tensor = transform(image).unsqueeze(0)

# Inférence
with torch.no_grad():
    outputs = model(image_tensor)
    cls_scores = outputs['cls_scores']
    bbox_preds = outputs['bbox_preds']

# Post-traitement pour extraire les détections
# (voir inference.py pour le code complet)
```

## 📊 Performance

- **Dataset** : {config.get('dataset_description', 'Custom aircraft parts dataset')}
- **Epochs** : {config.get('epochs', 'N/A')}
- **Batch size** : {config.get('batch_size', 'N/A')}
- **Learning rate** : {config.get('learning_rate', 'N/A')}

## 📁 Structure du modèle

```
.
├── internvl2/              # Modèle InternVL2-2B complet
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── detection_weights.pt    # Poids de la tête de détection
├── config.yaml            # Configuration d'entraînement
└── README.md              # Ce fichier
```

## 🔧 Configuration

```yaml
{yaml.dump(config, default_flow_style=False, allow_unicode=True)}
```

## 📝 Citation

Si vous utilisez ce modèle, merci de citer :

```bibtex
@misc{{internvl2-aircraft-detection,
  title={{InternVL2-2B Aircraft Parts Detection}},
  author={{Your Name}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/{repo_name}}}}}
}}
```

## 📄 Licence

MIT License - Voir le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

- **InternVL2** par OpenGVLab : [OpenGVLab/InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B)
- Basé sur l'architecture Vision-Language Model

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur le repo.

---

**Note** : Ce modèle contient le modèle InternVL2-2B complet (~4.4GB). Si vous voulez seulement les poids de la tête de détection (~10MB), utilisez le checkpoint léger `checkpoint_best.pt`.
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Model card créé : {output_path}")


def push_full_model_to_hf(
    full_model_dir,
    repo_name,
    hf_token=None,
    private=True,
    commit_message="Upload complete InternVL2-2B detection model"
):
    """
    Push le modèle COMPLET sur Hugging Face Hub
    
    Args:
        full_model_dir: Chemin vers le dossier full_model_best
        repo_name: Nom du repo HF
        hf_token: Token HF
        private: Repo privé ou public
        commit_message: Message de commit
    """
    
    print(f"🚀 Push du MODÈLE COMPLET vers Hugging Face Hub...")
    print(f"📦 Dossier: {full_model_dir}")
    print(f"🏷️  Repo: {repo_name}")
    
    full_model_path = Path(full_model_dir)
    
    # Vérifie que le dossier existe
    if not full_model_path.exists():
        raise FileNotFoundError(f"Dossier introuvable: {full_model_dir}")
    
    # Récupère le token
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            raise ValueError("Token Hugging Face requis. Utilisez --token ou HF_TOKEN env var")
    
    # API Hugging Face
    api = HfApi(token=hf_token)
    
    # Crée le repo (si n'existe pas)
    print(f"📁 Création du repo '{repo_name}' (si nécessaire)...")
    try:
        create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            exist_ok=True
        )
        print("✅ Repo créé/trouvé")
    except Exception as e:
        print(f"⚠️  Erreur création repo: {e}")
    
    # Crée le README.md
    config_path = full_model_path.parent.parent / "config.yaml"
    readme_path = full_model_path / "README.md"
    
    if config_path.exists():
        create_model_card(repo_name, config_path, readme_path)
    else:
        print("⚠️  config.yaml introuvable, README.md minimal créé")
        with open(readme_path, 'w') as f:
            f.write(f"# {repo_name}\n\nModèle InternVL2-2B fine-tuné pour la détection d'objets.")
    
    print(f"\n📤 Upload des fichiers (cela peut prendre plusieurs minutes pour ~4.4GB)...")
    print("☕ Allez vous chercher un café, ça va prendre un peu de temps...")
    
    # Upload tout le dossier
    try:
        upload_folder(
            folder_path=str(full_model_path),
            repo_id=repo_name,
            token=hf_token,
            commit_message=commit_message,
            ignore_patterns=["*.pyc", "__pycache__", "*.git*"]
        )
        print(f"\n✅ Modèle complet uploadé avec succès!")
        print(f"🔗 Lien: https://huggingface.co/{repo_name}")
        
        # Instructions pour télécharger
        print(f"\n📥 Pour télécharger le modèle complet:")
        print(f"   from huggingface_hub import snapshot_download")
        print(f"   snapshot_download(repo_id='{repo_name}', local_dir='./my_model')")
        
    except Exception as e:
        print(f"❌ Erreur upload: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Push le MODÈLE COMPLET sur Hugging Face Hub")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Chemin vers le dossier full_model_best (ex: experiments/run_XXX/checkpoints/full_model_best)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Nom du repo HF (ex: 'yourusername/model-name')"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token Hugging Face (ou utilisez HF_TOKEN env var)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Rendre le repo public (par défaut: privé)"
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Upload complete InternVL2-2B detection model",
        help="Message de commit"
    )
    
    args = parser.parse_args()
    
    push_full_model_to_hf(
        full_model_dir=args.model_dir,
        repo_name=args.repo,
        hf_token=args.token,
        private=not args.public,
        commit_message=args.message
    )


if __name__ == "__main__":
    main()
