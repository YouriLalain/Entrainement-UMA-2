#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour pusher le modèle entraîné sur Hugging Face Hub
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
import torch

def push_model_to_hf(
    checkpoint_path,
    repo_name,
    hf_token=None,
    private=True,
    commit_message="Upload trained InternVL2-2B detection model"
):
    """
    Push le modèle sur Hugging Face Hub
    
    Args:
        checkpoint_path: Chemin vers le checkpoint (ex: experiments/run_001/checkpoints/checkpoint_best.pt)
        repo_name: Nom du repo HF (ex: "yourilalain/internvl2-detection-aircraft")
        hf_token: Token HF (ou None pour utiliser la variable d'environnement)
        private: Repo privé ou public
        commit_message: Message de commit
    """
    
    print(f"🚀 Push du modèle vers Hugging Face Hub...")
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"🏷️  Repo: {repo_name}")
    
    # Vérifie que le checkpoint existe
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
    
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
    
    # Prépare le dossier temporaire avec les fichiers à upload
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))  # experiments/run_XXX
    
    print(f"\n📤 Upload des fichiers depuis {run_dir}...")
    
    # Upload tout le dossier du run
    try:
        upload_folder(
            folder_path=run_dir,
            repo_id=repo_name,
            token=hf_token,
            commit_message=commit_message,
            ignore_patterns=["*.pyc", "__pycache__", "*.git*", "logs/*"]  # Ignore les fichiers inutiles
        )
        print(f"✅ Modèle uploadé avec succès!")
        print(f"🔗 Lien: https://huggingface.co/{repo_name}")
        
        # Instructions pour télécharger
        print(f"\n📥 Pour télécharger le modèle:")
        print(f"   from huggingface_hub import snapshot_download")
        print(f"   snapshot_download(repo_id='{repo_name}', local_dir='./model')")
        
    except Exception as e:
        print(f"❌ Erreur upload: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Push un modèle entraîné sur Hugging Face Hub")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin vers le checkpoint (ex: experiments/run_001_2025-10-29_07-55/checkpoints/checkpoint_best.pt)"
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
        default="Upload trained InternVL2-2B detection model",
        help="Message de commit"
    )
    
    args = parser.parse_args()
    
    push_model_to_hf(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo,
        hf_token=args.token,
        private=not args.public,
        commit_message=args.message
    )


if __name__ == "__main__":
    main()
