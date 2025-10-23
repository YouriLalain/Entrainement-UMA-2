#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'inférence pour tester le modèle entraîné
"""

import torch
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import argparse

from model import load_model, load_checkpoint


# Classes
CLASSES = [
    "fuselage", "cockpit", "derive", "empennage", "aile", "aileron",
    "volet", "reacteur", "nacelle_moteur", "tuyere", "train_atterrissage", "porte"
]


def load_inference_model(checkpoint_path, config_path='config.yaml'):
    """Charge le modèle depuis un checkpoint"""
    
    # Charge la config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Crée le modèle
    device = torch.device(config['device'])
    model = load_model(config)
    
    # Charge le checkpoint
    load_checkpoint(model, checkpoint_path, device)
    
    model.eval()
    
    return model, device, config


def preprocess_image(image_path, size=(384, 384)):
    """Préprocess l'image pour l'inférence"""
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image, original_size


def postprocess_predictions(cls_scores, bbox_preds, conf_threshold=0.5, nms_threshold=0.5):
    """Post-traite les prédictions"""
    
    # cls_scores: [1, num_classes, H, W]
    # bbox_preds: [1, 4, H, W]
    
    batch_size, num_classes, H, W = cls_scores.shape
    
    # Trouve les détections avec score > threshold
    cls_probs = torch.softmax(cls_scores, dim=1)
    max_probs, max_classes = cls_probs.max(dim=1)  # [1, H, W]
    
    # Seuillage
    mask = max_probs[0] > conf_threshold
    
    if not mask.any():
        return []
    
    # Récupère les positions
    y_coords, x_coords = torch.where(mask)
    
    detections = []
    for y, x in zip(y_coords, x_coords):
        class_id = max_classes[0, y, x].item()
        confidence = max_probs[0, y, x].item()
        
        # Récupère la bbox
        bbox = bbox_preds[0, :, y, x].cpu().numpy()  # [4]
        
        detections.append({
            'class_id': class_id,
            'class_name': CLASSES[class_id] if class_id < len(CLASSES) else 'unknown',
            'confidence': confidence,
            'bbox': bbox.tolist()  # [x, y, w, h] normalisé
        })
    
    # Simple NMS (garde les top-K par classe)
    # Version simplifiée
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:10]
    
    return detections


def visualize_detections(image, detections, output_path=None):
    """Visualise les détections sur l'image"""
    
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Couleurs par classe
    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
        '#800000', '#008000', '#000080', '#808000', '#800080', '#008080'
    ]
    
    for det in detections:
        x, y, w, h = det['bbox']
        
        # Dénormalise
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
        
        # Couleur
        color = colors[det['class_id'] % len(colors)]
        
        # Draw bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{det['class_name']} {det['confidence']:.2f}"
        
        # Background pour le texte
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        bbox_text = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1), label, fill='white', font=font)
    
    if output_path:
        image.save(output_path)
        print(f"💾 Image sauvegardée : {output_path}")
    
    return image


def generate_description(detections):
    """Génère une description textuelle des détections"""
    
    if not detections:
        return "Aucun composant détecté."
    
    # Groupe par classe
    class_counts = {}
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Génère la description
    components = []
    for class_name, count in class_counts.items():
        if count == 1:
            components.append(class_name)
        else:
            components.append(f"{count} {class_name}s")
    
    if len(components) == 1:
        desc = f"Je détecte 1 composant : {components[0]}"
    else:
        desc = f"Je détecte {len(detections)} composants : {', '.join(components[:-1])} et {components[-1]}"
    
    return desc


def inference(image_path, checkpoint_path, config_path='config.yaml', output_path=None, conf_threshold=0.5):
    """Inférence sur une image"""
    
    print("="*70)
    print("🔍 INFERENCE OVIS DETECTION")
    print("="*70)
    
    # Charge le modèle
    print("🔄 Chargement du modèle...")
    model, device, config = load_inference_model(checkpoint_path, config_path)
    print("✅ Modèle chargé")
    
    # Charge et préprocess l'image
    print(f"📷 Chargement de l'image : {image_path}")
    image_tensor, original_image, original_size = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Inférence
    print("🚀 Inférence...")
    with torch.no_grad():
        outputs = model(image_tensor)
        cls_scores = outputs['cls_scores']
        bbox_preds = outputs['bbox_preds']
    
    # Post-traitement
    print("🔧 Post-traitement...")
    detections = postprocess_predictions(cls_scores, bbox_preds, conf_threshold)
    
    # Affiche les résultats
    print("\n" + "="*70)
    print("📊 RÉSULTATS")
    print("="*70)
    
    if not detections:
        print("❌ Aucune détection")
    else:
        print(f"✅ {len(detections)} détection(s)\n")
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det['class_name']}")
            print(f"   - Confiance: {det['confidence']:.3f}")
            print(f"   - BBox: {det['bbox']}\n")
    
    # Génère la description
    description = generate_description(detections)
    print("💬 DESCRIPTION")
    print(f"   {description}\n")
    
    # Visualise
    if output_path or detections:
        print("🎨 Visualisation...")
        vis_image = original_image.copy()
        vis_image = visualize_detections(vis_image, detections, output_path)
        
        if not output_path:
            vis_image.show()
    
    print("="*70)
    print("✅ TERMINÉ")
    print("="*70)
    
    return detections, description


def main():
    parser = argparse.ArgumentParser(description='Inférence OVIS Detection')
    parser.add_argument('--image', type=str, required=True, help='Chemin vers l\'image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Chemin vers le checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Chemin vers la config')
    parser.add_argument('--output', type=str, default=None, help='Chemin de sortie pour l\'image avec détections')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Seuil de confiance')
    
    args = parser.parse_args()
    
    inference(
        args.image,
        args.checkpoint,
        args.config,
        args.output,
        args.conf_threshold
    )


if __name__ == "__main__":
    main()
