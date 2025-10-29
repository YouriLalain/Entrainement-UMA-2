#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour exporter le modèle entraîné (zip, tar.gz, etc.)
"""

import os
import argparse
import shutil
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime


def create_zip(source_dir, output_path):
    """Crée un zip du dossier"""
    print(f"📦 Création du ZIP: {output_path}")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Ignore les logs trop volumineux
            if 'logs' in dirs:
                dirs.remove('logs')
            
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                zipf.write(file_path, arcname)
                print(f"   + {arcname}")
    
    # Taille du fichier
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ ZIP créé: {output_path} ({size_mb:.2f} MB)")


def create_tar_gz(source_dir, output_path):
    """Crée un tar.gz du dossier"""
    print(f"📦 Création du TAR.GZ: {output_path}")
    
    with tarfile.open(output_path, 'w:gz') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    
    # Taille du fichier
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ TAR.GZ créé: {output_path} ({size_mb:.2f} MB)")


def export_model(run_dir, output_dir="exports", format="zip", include_logs=False):
    """
    Exporte le modèle dans un format compressé
    
    Args:
        run_dir: Dossier du run (ex: experiments/run_001_2025-10-29_07-55)
        output_dir: Dossier de sortie
        format: Format de compression (zip, tar.gz)
        include_logs: Inclure les logs (peut être volumineux)
    """
    
    print(f"🚀 Export du modèle...")
    print(f"📁 Source: {run_dir}")
    print(f"📦 Format: {format}")
    
    # Vérifie que le run existe
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory introuvable: {run_dir}")
    
    # Crée le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Nom du fichier de sortie
    run_name = os.path.basename(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "zip":
        output_path = os.path.join(output_dir, f"{run_name}_{timestamp}.zip")
        create_zip(run_dir, output_path)
    elif format == "tar.gz":
        output_path = os.path.join(output_dir, f"{run_name}_{timestamp}.tar.gz")
        create_tar_gz(run_dir, output_path)
    else:
        raise ValueError(f"Format non supporté: {format}. Utilisez 'zip' ou 'tar.gz'")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Exporter un modèle entraîné")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Dossier du run (ex: experiments/run_001_2025-10-29_07-55)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Dossier de sortie (défaut: exports)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["zip", "tar.gz"],
        default="zip",
        help="Format de compression (défaut: zip)"
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Inclure les logs (peut être volumineux)"
    )
    
    args = parser.parse_args()
    
    output_path = export_model(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        format=args.format,
        include_logs=args.include_logs
    )
    
    print(f"\n🎉 Export terminé!")
    print(f"📦 Fichier: {output_path}")
    print(f"📥 Vous pouvez maintenant télécharger ce fichier depuis RunPod")


if __name__ == "__main__":
    main()
