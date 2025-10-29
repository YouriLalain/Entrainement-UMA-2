# 📦 Guide d'Export du Modèle

## 🎯 Deux options d'export

### 1️⃣ **Modèle COMPLET** (InternVL2-2B + tête) - 4.4GB
✅ Utilisable directement sans téléchargement supplémentaire  
✅ Modèle "clé en main"  
❌ Upload/download plus long (~4.4GB)

### 2️⃣ **Tête uniquement** - 10MB
✅ Upload/download ultra-rapide  
❌ Nécessite de télécharger InternVL2-2B séparément  

---

## 🚀 Option 1 : Push du MODÈLE COMPLET (Recommandé pour vous)

### Étape 1 : L'entraînement sauvegarde automatiquement

Quand le meilleur modèle est sauvegardé, le système crée automatiquement :

```
experiments/run_XXX/checkpoints/
├── checkpoint_best.pt          # Tête uniquement (10MB)
└── full_model_best/            # MODÈLE COMPLET (4.4GB) ✨
    ├── internvl2/              # InternVL2-2B complet
    │   ├── config.json
    │   ├── model.safetensors
    │   └── ...
    └── detection_weights.pt    # Votre tête de détection
```

### Étape 2 : Push sur Hugging Face

```bash
# Sur RunPod, après l'entraînement
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

python push_full_model_to_hf.py \
    --model-dir experiments/run_003_2025-10-29_07-57/checkpoints/full_model_best \
    --repo "VotreUsername/internvl2-aircraft-detection"

# Pour un repo public
python push_full_model_to_hf.py \
    --model-dir experiments/run_003_2025-10-29_07-57/checkpoints/full_model_best \
    --repo "VotreUsername/internvl2-aircraft-detection" \
    --public
```

⏱️ **Temps estimé** : 5-10 minutes pour upload 4.4GB (avec bonne connexion)

### Étape 3 : Utilisation (sur votre Mac)

```bash
# Installation
pip install torch transformers huggingface_hub pillow pyyaml

# Téléchargement
huggingface-cli download VotreUsername/internvl2-aircraft-detection --local-dir ./my_model
```

```python
# Utilisation
from huggingface_hub import snapshot_download
import torch
from model import load_model
import yaml

# Télécharge le modèle complet
model_path = snapshot_download(
    repo_id="VotreUsername/internvl2-aircraft-detection",
    local_dir="./my_model"
)

# Charge la config
with open("my_model/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Charge le modèle (tout est déjà là !)
model = load_model(config)
checkpoint = torch.load("my_model/detection_weights.pt")
model.detection_head.load_state_dict(checkpoint['detection_head'])
model.roi_projector.load_state_dict(checkpoint['roi_projector'])

# Prêt à l'emploi ! 🚀
```

---

## � Option 2 : Tête uniquement (Alternative légère)

### Obtenir votre token Hugging Face

1. Allez sur https://huggingface.co/settings/tokens
2. Cliquez sur "New token"
3. Donnez-lui un nom (ex: "runpod-upload")
4. Sélectionnez "Write" pour les permissions
5. Copiez le token

### Sur RunPod, configurez le token :

```bash
# Méthode 1 : Variable d'environnement (recommandé)
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc

# Méthode 2 : Utiliser --token dans la commande
```

---

## 📤 Méthode 1 : Push sur Hugging Face Hub

### Installation des dépendances

```bash
pip install huggingface_hub
```

### Utilisation

```bash
# Exemple : pusher le meilleur checkpoint
python push_to_hf.py \
    --checkpoint experiments/run_001_2025-10-29_07-55/checkpoints/checkpoint_best.pt \
    --repo "yourilalain/internvl2-aircraft-detection" \
    --message "InternVL2-2B fine-tuned for aircraft parts detection"

# Pour un repo public
python push_to_hf.py \
    --checkpoint experiments/run_001_2025-10-29_07-55/checkpoints/checkpoint_best.pt \
    --repo "yourilalain/internvl2-aircraft-detection" \
    --public

# Avec token en argument (si pas dans env var)
python push_to_hf.py \
    --checkpoint experiments/run_001_2025-10-29_07-55/checkpoints/checkpoint_best.pt \
    --repo "yourilalain/internvl2-aircraft-detection" \
    --token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Télécharger depuis Hugging Face (sur votre Mac)

```python
from huggingface_hub import snapshot_download

# Télécharge tout le modèle
snapshot_download(
    repo_id="yourilalain/internvl2-aircraft-detection",
    local_dir="./my_model"
)
```

Ou en ligne de commande :

```bash
huggingface-cli download yourilalain/internvl2-aircraft-detection --local-dir ./my_model
```

---

## 📦 Méthode 2 : Export en ZIP

### Créer un ZIP du run

```bash
# Export en ZIP (recommandé, plus rapide)
python export_model.py \
    --run-dir experiments/run_001_2025-10-29_07-55 \
    --format zip

# Export en TAR.GZ (plus compressé)
python export_model.py \
    --run-dir experiments/run_001_2025-10-29_07-55 \
    --format tar.gz

# Spécifier le dossier de sortie
python export_model.py \
    --run-dir experiments/run_001_2025-10-29_07-55 \
    --output-dir /workspace/exports \
    --format zip
```

Le fichier sera créé dans `exports/run_001_2025-10-29_07-55_TIMESTAMP.zip`

### Télécharger depuis RunPod

#### Méthode A : Interface Web RunPod
1. Ouvrez le "File Browser" dans RunPod
2. Naviguez vers `exports/`
3. Clic droit sur le fichier → Download

#### Méthode B : SCP (depuis votre Mac)

```bash
# Récupérez l'IP et le port SSH de votre pod RunPod
# Puis téléchargez :
scp -P PORT root@POD_IP:/workspace/exports/run_001_*.zip ~/Downloads/
```

#### Méthode C : Servir via HTTP temporairement

Sur RunPod :
```bash
cd /workspace/exports
python -m http.server 8000
```

Puis dans RunPod, exposez le port 8000 et téléchargez via navigateur.

---

## 🚀 Commandes complètes (une seule ligne)

### Export ZIP + Download automatique

```bash
# Sur RunPod
python export_model.py \
    --run-dir experiments/run_001_2025-10-29_07-55 \
    --format zip && \
ls -lh exports/

# Notez le nom du fichier, puis téléchargez via File Browser
```

### Push sur HF (plus simple)

```bash
# Sur RunPod
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

python push_to_hf.py \
    --checkpoint experiments/run_001_2025-10-29_07-55/checkpoints/checkpoint_best.pt \
    --repo "yourilalain/internvl2-aircraft-detection"

# Sur votre Mac
pip install huggingface_hub
huggingface-cli download yourilalain/internvl2-aircraft-detection --local-dir ./my_model
```

---

## 📊 Comparaison des méthodes

| Critère | Hugging Face Hub | ZIP/TAR.GZ |
|---------|------------------|------------|
| **Vitesse upload** | ⚡⚡⚡ Très rapide | ⚡⚡ Moyen |
| **Vitesse download** | ⚡⚡⚡ Très rapide | ⚡⚡ Moyen |
| **Partage** | ✅ Facile | ❌ Difficile |
| **Versioning** | ✅ Automatique | ❌ Manuel |
| **Simplicité** | ✅ Simple | ⚠️ Nécessite SCP/HTTP |
| **Taille limite** | 🔥 Illimitée (LFS) | ⚠️ Selon RunPod |

**Recommandation** : 🏆 **Hugging Face Hub** pour la plupart des cas.

---

## 🔧 Troubleshooting

### Erreur "Token invalid"
```bash
# Vérifiez que le token est bien configuré
echo $HF_TOKEN

# Ou utilisez --token directement
python push_to_hf.py --token "hf_xxx..." ...
```

### Fichier trop volumineux
```bash
# Excluez les logs
python export_model.py \
    --run-dir experiments/run_001_2025-10-29_07-55 \
    --format zip
# (les logs sont déjà exclus par défaut)
```

### Upload lent sur HF
```bash
# Activez hf_transfer pour upload rapide
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## ✅ Checklist finale

Avant de terminer votre session RunPod :

- [ ] Entraînement terminé
- [ ] Meilleur checkpoint identifié (`checkpoint_best.pt`)
- [ ] Token HF configuré
- [ ] Modèle pushé sur HF **OU** ZIP créé
- [ ] Modèle téléchargé sur votre Mac
- [ ] Session RunPod terminée 💰

---

## 📝 Exemple complet

```bash
# 1. Entraînement terminé, identifier le meilleur run
ls -lt experiments/

# 2. Exporter (choisissez UNE méthode)

# Option A : Hugging Face (recommandé)
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python push_to_hf.py \
    --checkpoint experiments/run_001_2025-10-29_07-55/checkpoints/checkpoint_best.pt \
    --repo "yourilalain/internvl2-aircraft-detection"

# Option B : ZIP
python export_model.py \
    --run-dir experiments/run_001_2025-10-29_07-55 \
    --format zip

# 3. Sur votre Mac, télécharger
huggingface-cli download yourilalain/internvl2-aircraft-detection --local-dir ./my_model

# 4. Tester l'inférence
python inference.py \
    --image dataset/images/test.jpg \
    --checkpoint my_model/checkpoints/checkpoint_best.pt \
    --output result.jpg
```

🎉 **C'est tout !**
