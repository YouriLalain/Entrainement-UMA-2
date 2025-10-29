#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test interactif du modèle InternVL2 Detection
Usage: python test_interactive.py --image path/to/image.jpg --question "Quelles sont les parties de l'avion ?"
"""

import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import yaml
import numpy as np
from pathlib import Path

from model import load_model, load_checkpoint


# Couleurs pour chaque classe
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#F67280',
    '#C06C84', '#6C5B7B'
]


def load_image(image_path, image_size=448):
    """Charge et prépare une image"""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    # Transform pour le modèle
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, 448, 448]
    
    # Image pour visualisation
    display_image = image.resize((image_size, image_size))
    
    return image_tensor, display_image, original_size


def decode_predictions(cls_scores, bbox_preds, confidence_threshold=0.3, top_k=10):
    """
    Décode les prédictions du modèle
    
    Args:
        cls_scores: [B, num_classes, H, W]
        bbox_preds: [B, 4, H, W]
        confidence_threshold: Seuil de confiance minimum
        top_k: Nombre maximum de détections
        
    Returns:
        List of (class_id, confidence, bbox)
    """
    B, num_classes, H, W = cls_scores.shape
    
    # Applique softmax sur les classes
    cls_probs = torch.softmax(cls_scores, dim=1)  # [B, num_classes, H, W]
    
    # Trouve les meilleures prédictions
    max_probs, class_ids = torch.max(cls_probs, dim=1)  # [B, H, W]
    
    # Flatten pour traiter
    max_probs_flat = max_probs.view(B, -1)  # [B, H*W]
    class_ids_flat = class_ids.view(B, -1)  # [B, H*W]
    bbox_preds_flat = bbox_preds.view(B, 4, -1)  # [B, 4, H*W]
    
    detections = []
    
    for b in range(B):
        # Filtre par confiance
        mask = max_probs_flat[b] > confidence_threshold
        
        if mask.sum() == 0:
            continue
        
        confidences = max_probs_flat[b][mask]
        classes = class_ids_flat[b][mask]
        bboxes = bbox_preds_flat[b][:, mask].T  # [N, 4]
        
        # Trie par confiance décroissante
        sorted_indices = torch.argsort(confidences, descending=True)[:top_k]
        
        for idx in sorted_indices:
            class_id = classes[idx].item()
            confidence = confidences[idx].item()
            bbox = bboxes[idx].cpu().numpy()  # [x, y, w, h] normalisé [0, 1]
            
            detections.append({
                'class_id': class_id,
                'class_name': None,  # Sera rempli après
                'confidence': confidence,
                'bbox': bbox
            })
    
    return detections


def draw_detections(image, detections, class_names, min_confidence=0.3):
    """Dessine les détections sur l'image"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Police (fallback si pas disponible)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    print(f"\n🎯 Détections (confiance > {min_confidence}):")
    print("=" * 80)
    
    for i, det in enumerate(detections):
        if det['confidence'] < min_confidence:
            continue
        
        class_id = det['class_id']
        class_name = class_names[class_id]
        confidence = det['confidence']
        bbox = det['bbox']  # [x, y, w, h] normalisé
        
        # Convertit en coordonnées pixels
        x = bbox[0] * width
        y = bbox[1] * height
        w = bbox[2] * width
        h = bbox[3] * height
        
        # Coordonnées du rectangle
        x1, y1 = x - w/2, y - h/2
        x2, y2 = x + w/2, y + h/2
        
        # Couleur de la classe
        color = COLORS[class_id % len(COLORS)]
        
        # Dessine le rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Label
        label = f"{class_name} ({confidence:.2%})"
        
        # Background pour le texte
        bbox_text = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 20), label, fill='white', font=font)
        
        print(f"  {i+1}. {class_name:20s} | Confiance: {confidence:6.2%} | "
              f"BBox: x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}")
    
    print("=" * 80)
    
    return image


def generate_text_response(detections, class_names, question):
    """Génère une réponse textuelle basée sur les détections"""
    
    if not detections:
        return "❌ Aucune partie d'avion détectée dans cette image."
    
    # Groupe par classe
    detected_parts = {}
    for det in detections:
        class_name = class_names[det['class_id']]
        if class_name not in detected_parts:
            detected_parts[class_name] = []
        detected_parts[class_name].append(det['confidence'])
    
    # Statistiques
    response = f"✅ J'ai détecté {len(detections)} partie(s) d'avion:\n\n"
    
    for class_name, confidences in sorted(detected_parts.items(), 
                                          key=lambda x: max(x[1]), 
                                          reverse=True):
        count = len(confidences)
        avg_conf = np.mean(confidences)
        max_conf = max(confidences)
        
        response += f"  • {class_name}: {count} occurrence(s) "
        response += f"(confiance max: {max_conf:.1%}, moyenne: {avg_conf:.1%})\n"
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Test interactif du modèle InternVL2 Detection")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image")
    parser.add_argument("--question", type=str, default="Quelles parties de l'avion vois-tu ?", 
                       help="Question à poser au modèle")
    parser.add_argument("--checkpoint", type=str, 
                       default="experiments/run_003_2025-10-29_07-57/checkpoints/checkpoint_best.pt",
                       help="Chemin vers le checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Chemin de sortie pour l'image (optionnel)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Seuil de confiance minimum")
    parser.add_argument("--top-k", type=int, default=20, help="Nombre maximum de détections")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 TEST INTERACTIF - InternVL2 Detection")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Charge la config
    print(f"📋 Chargement de la config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['classes']
    print(f"📊 Classes: {len(class_names)}")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    
    # Charge le modèle
    print(f"\n🔄 Chargement du modèle...")
    model = load_model(config)
    
    # Charge le checkpoint
    if Path(args.checkpoint).exists():
        load_checkpoint(model, args.checkpoint, device)
    else:
        print(f"⚠️  Checkpoint introuvable: {args.checkpoint}")
        print("   Utilisation du modèle non entraîné (pour test)")
    
    model.eval()
    
    # Charge l'image
    print(f"\n🖼️  Chargement de l'image: {args.image}")
    image_tensor, display_image, original_size = load_image(args.image)
    image_tensor = image_tensor.to(device)
    
    print(f"   Taille originale: {original_size}")
    print(f"   Taille modèle: {image_tensor.shape}")
    
    # Question
    print(f"\n❓ Question: \"{args.question}\"")
    
    # Inférence
    print(f"\n🔮 Inférence...")
    with torch.no_grad():
        outputs = model(image_tensor)
        cls_scores = outputs['cls_scores']
        bbox_preds = outputs['bbox_preds']
    
    # Décode les prédictions
    print(f"📦 Décodage des prédictions...")
    detections = decode_predictions(
        cls_scores, 
        bbox_preds, 
        confidence_threshold=args.confidence,
        top_k=args.top_k
    )
    
    # Ajoute les noms de classes
    for det in detections:
        det['class_name'] = class_names[det['class_id']]
    
    # Génère la réponse textuelle
    print(f"\n💬 Réponse:")
    print("-" * 80)
    response = generate_text_response(detections, class_names, args.question)
    print(response)
    print("-" * 80)
    
    # Dessine les détections
    result_image = draw_detections(
        display_image.copy(), 
        detections, 
        class_names,
        min_confidence=args.confidence
    )
    
    # Sauvegarde l'image
    if args.output:
        output_path = args.output
    else:
        # Génère un nom automatique
        image_name = Path(args.image).stem
        output_path = f"result_{image_name}.jpg"
    
    result_image.save(output_path)
    print(f"\n💾 Image sauvegardée: {output_path}")
    
    # Affiche un résumé
    print(f"\n📈 Résumé:")
    print(f"   • Détections totales: {len(detections)}")
    print(f"   • Confiance moyenne: {np.mean([d['confidence'] for d in detections]):.1%}")
    print(f"   • Classes détectées: {len(set(d['class_id'] for d in detections))}/{len(class_names)}")
    
    print("\n" + "=" * 80)
    print("✅ Test terminé!")
    print("=" * 80)


if __name__ == "__main__":
    main()
