#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement avec gestion automatique des versions (run_001, run_002, ...)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import yaml
from datetime import datetime
import shutil
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import re

from model import load_model


class DetectionDataset(Dataset):
    """Dataset pour l'entraînement"""
    
    def __init__(self, json_path, images_dir, image_size=448, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.transform = transform or self.default_transform()
    
    def default_transform(self):
        """Transform par défaut - InternVL2-2B utilise 448x448"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Charge l'image
        img_path = self.images_dir / item['image']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Prépare les bounding boxes
        bboxes = []
        labels = []
        for bbox_info in item['bboxes']:
            bboxes.append(bbox_info['bbox'])  # [x, y, w, h] normalisé
            labels.append(bbox_info['label'])
        
        return {
            'image': image,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': labels,
            'conversations': item['conversations']
        }


def collate_fn(batch):
    """Collate function pour le DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    bboxes = [item['bboxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    conversations = [item['conversations'] for item in batch]
    
    return {
        'images': images,
        'bboxes': bboxes,
        'labels': labels,
        'conversations': conversations
    }


class DetectionLoss(nn.Module):
    """Loss pour la détection"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()
    
    def forward(self, cls_scores, bbox_preds, target_bboxes, target_labels):
        """
        Calcule la loss (version simplifiée)
        
        Dans une vraie implémentation, tu ferais du matching entre
        prédictions et targets (Hungarian matching, etc.)
        """
        # Simplifié pour l'exemple
        # cls_scores: [B, num_classes, H, W]
        # bbox_preds: [B, 4, H, W]
        
        # Pour l'instant, on fait une loss simple sur les scores moyens
        batch_size = cls_scores.size(0)
        
        # Loss de classification (moyenne sur toute la carte)
        cls_loss = cls_scores.mean() * 0.1  # Placeholder
        
        # Loss de bbox (moyenne)
        bbox_loss = bbox_preds.mean() * 0.1  # Placeholder
        
        total_loss = cls_loss + bbox_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss
        }


def get_next_run_number(experiments_dir):
    """Trouve le prochain numéro de run disponible"""
    experiments_path = Path(experiments_dir)
    experiments_path.mkdir(parents=True, exist_ok=True)
    
    existing_runs = list(experiments_path.glob("run_*"))
    
    if not existing_runs:
        return 1
    
    # Extrait les numéros
    numbers = []
    for run_dir in existing_runs:
        match = re.match(r"run_(\d+)", run_dir.name)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) + 1 if numbers else 1


def create_run_directory(base_dir="experiments"):
    """Crée un nouveau dossier de run avec timestamp"""
    run_number = get_next_run_number(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"run_{run_number:03d}_{timestamp}"
    
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Crée les sous-dossiers
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    
    print(f"📁 Run directory créé : {run_dir}")
    
    return run_dir


def save_config(config, run_dir):
    """Sauvegarde la config dans le dossier de run"""
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 Config sauvegardée : {config_path}")


def save_checkpoint(model, optimizer, epoch, loss, run_dir, is_best=False):
    """Sauvegarde un checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'detection_head': model.detection_head.state_dict(),
        'roi_projector': model.roi_projector.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    
    # Sauvegarde le checkpoint (lightweight - seulement la tête)
    checkpoint_path = run_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Sauvegarde aussi le dernier
    last_path = run_dir / "checkpoints" / "checkpoint_last.pt"
    torch.save(checkpoint, last_path)
    
    # Si c'est le meilleur, sauvegarde aussi
    if is_best:
        best_path = run_dir / "checkpoints" / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"🌟 Meilleur checkpoint sauvegardé (loss: {loss:.4f})")
        
        # Sauvegarde aussi le modèle COMPLET (pour push HF)
        print("💾 Sauvegarde du modèle complet...")
        full_model_path = run_dir / "checkpoints" / "full_model_best"
        model.internvl_model.save_pretrained(full_model_path / "internvl2")
        model.tokenizer.save_pretrained(full_model_path / "internvl2")
        
        # Sauvegarde la tête aussi dans le même dossier
        torch.save(checkpoint, full_model_path / "detection_weights.pt")
        print(f"✅ Modèle complet sauvegardé dans {full_model_path}")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Entraîne une epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        target_bboxes = batch['bboxes']
        target_labels = batch['labels']
        
        # Forward
        outputs = model(images)
        cls_scores = outputs['cls_scores']
        bbox_preds = outputs['bbox_preds']
        
        # Calcule la loss
        loss_dict = criterion(cls_scores, bbox_preds, target_bboxes, target_labels)
        loss = loss_dict['total_loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_cls_loss += loss_dict['cls_loss'].item()
        total_bbox_loss += loss_dict['bbox_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{loss_dict['cls_loss'].item():.4f}",
            'bbox': f"{loss_dict['bbox_loss'].item():.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_bbox_loss = total_bbox_loss / len(dataloader)
    
    return avg_loss, avg_cls_loss, avg_bbox_loss


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['images'].to(device)
            target_bboxes = batch['bboxes']
            target_labels = batch['labels']
            
            # Forward
            outputs = model(images)
            cls_scores = outputs['cls_scores']
            bbox_preds = outputs['bbox_preds']
            
            # Loss
            loss_dict = criterion(cls_scores, bbox_preds, target_bboxes, target_labels)
            
            total_loss += loss_dict['total_loss'].item()
            total_cls_loss += loss_dict['cls_loss'].item()
            total_bbox_loss += loss_dict['bbox_loss'].item()
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_bbox_loss = total_bbox_loss / len(dataloader)
    
    return avg_loss, avg_cls_loss, avg_bbox_loss


def train(config):
    """Fonction principale d'entraînement"""
    
    print("="*70)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT OVIS DETECTION")
    print("="*70)
    
    # Crée le dossier de run
    run_dir = create_run_directory(config['experiments_dir'])
    
    # Sauvegarde la config
    save_config(config, run_dir)
    
    # Device
    device = torch.device(config['device'])
    print(f"🖥️  Device: {device}")
    
    # Charge le modèle
    print("\n" + "="*70)
    model = load_model(config)
    model = model.to(device)  # Déplace explicitement le modèle sur le device
    print(f"✅ Modèle déplacé vers {device}")
    
    # Dataset
    print("\n" + "="*70)
    print("📂 CHARGEMENT DES DONNÉES")
    print("="*70)
    
    train_dataset = DetectionDataset(
        config['train_json'],
        config['images_dir']
    )
    val_dataset = DetectionDataset(
        config['val_json'],
        config['images_dir']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    print(f"✅ Train: {len(train_dataset)} samples")
    print(f"✅ Val: {len(val_dataset)} samples")
    
    # Loss et optimizer
    criterion = DetectionLoss(config['num_classes'])
    
    # Optimise seulement les paramètres trainables
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    print("\n" + "="*70)
    print("🏋️  ENTRAÎNEMENT")
    print("="*70)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n📅 Epoch {epoch}/{config['epochs']}")
        
        # Train
        train_loss, train_cls, train_bbox = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"   Train - Loss: {train_loss:.4f}, Cls: {train_cls:.4f}, BBox: {train_bbox:.4f}")
        
        # Validation
        val_loss, val_cls, val_bbox = validate(
            model, val_loader, criterion, device
        )
        
        print(f"   Val   - Loss: {val_loss:.4f}, Cls: {val_cls:.4f}, BBox: {val_bbox:.4f}")
        
        # Sauvegarde le checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % config['save_every'] == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, run_dir, is_best)
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*70)
    print(f"📁 Résultats dans : {run_dir}")
    print(f"🌟 Meilleure val loss : {best_val_loss:.4f}")


def main():
    # Charge la config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Lance l'entraînement
    train(config)


if __name__ == "__main__":
    main()
