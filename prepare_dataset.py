#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Préparation du dataset pour l'entraînement de détection d'objets avec Ovis
Génère 3 types de conversations par image
"""

import pandas as pd
import json
import os
from pathlib import Path
from collections import defaultdict
import random
from PIL import Image

# Configuration
CSV_PATH = "dataset/labels_my-project-name_2025-10-19-08-36-47.csv"
IMAGES_DIR = "dataset/images"
OUTPUT_DIR = "dataset/processed"
TRAIN_RATIO = 0.8

# Classes (12 labels)
CLASSES = [
    "fuselage", "cockpit", "derive", "empennage", "aile", "aileron",
    "volet", "reacteur", "nacelle_moteur", "tuyere", "train_atterrissage", "porte"
]

def load_and_group_annotations(csv_path):
    """Charge le CSV et groupe les annotations par image"""
    print(f"📂 Lecture du CSV : {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Groupe par image
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        img_name = row['image_name']
        grouped[img_name].append({
            'label': row['label_name'],
            'bbox': [
                float(row['bbox_x']),
                float(row['bbox_y']),
                float(row['bbox_width']),
                float(row['bbox_height'])
            ],
            'image_width': int(row['image_width']),
            'image_height': int(row['image_height'])
        })
    
    print(f"✅ {len(grouped)} images avec annotations")
    return grouped

def normalize_bbox(bbox, img_width, img_height):
    """Normalise les bbox en [0, 1]"""
    x, y, w, h = bbox
    return [
        x / img_width,
        y / img_height,
        w / img_width,
        h / img_height
    ]

def bbox_to_str(bbox):
    """Convertit bbox en string lisible"""
    x, y, w, h = bbox
    return f"[{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}]"

# Templates de questions variées
DETECTION_QUESTIONS = [
    "Détecte tous les composants de cet avion",
    "Quels sont les éléments visibles sur cet avion ?",
    "Liste tous les composants que tu vois",
    "Identifie toutes les parties de cet avion",
    "Analyse les composants de cet appareil",
    "Quelles parties de l'avion sont visibles ?",
    "Fais l'inventaire des composants visibles",
    "Repère tous les éléments de cet avion",
]

DESCRIPTION_QUESTIONS = [
    "Décris cet avion",
    "Que vois-tu sur cette image ?",
    "Présente cet appareil",
    "Donne-moi une description de cet avion",
    "Que peux-tu me dire sur cet avion ?",
    "Analyse cette image d'avion",
    "Décris les caractéristiques visibles",
    "Fais-moi un résumé de ce que tu vois",
]

LOCALIZATION_TEMPLATES = {
    "fuselage": ["Où est le fuselage ?", "Localise le fuselage", "Montre-moi le fuselage", "Position du fuselage ?"],
    "cockpit": ["Où est le cockpit ?", "Localise le cockpit", "Montre-moi le cockpit", "Où se trouve le poste de pilotage ?"],
    "derive": ["Où est la dérive ?", "Localise la dérive", "Montre-moi la dérive", "Position de la dérive ?"],
    "empennage": ["Où est l'empennage ?", "Localise l'empennage", "Montre-moi l'empennage", "Position de l'empennage ?"],
    "aile": ["Où sont les ailes ?", "Localise les ailes", "Montre-moi les ailes", "Position des ailes ?"],
    "aileron": ["Où sont les ailerons ?", "Localise les ailerons", "Montre-moi les ailerons", "Position des ailerons ?"],
    "volet": ["Où sont les volets ?", "Localise les volets", "Montre-moi les volets", "Position des volets ?"],
    "reacteur": ["Où sont les réacteurs ?", "Localise les moteurs", "Montre-moi les moteurs", "Où sont les moteurs ?", "Position des réacteurs ?"],
    "nacelle_moteur": ["Où sont les nacelles ?", "Localise les nacelles moteur", "Montre-moi les nacelles", "Position des nacelles ?"],
    "tuyere": ["Où sont les tuyères ?", "Localise les tuyères", "Montre-moi les tuyères", "Position des tuyères ?"],
    "train_atterrissage": ["Où est le train d'atterrissage ?", "Localise le train", "Montre-moi le train", "Position du train ?"],
    "porte": ["Où sont les portes ?", "Localise les portes", "Montre-moi les portes", "Position des portes ?"],
}

RESPONSE_TEMPLATES = {
    "single": [
        "Le {label} est situé à la position {bbox}",
        "Je vois le {label} à {bbox}",
        "Le {label} se trouve à {bbox}",
        "Voici le {label} : {bbox}",
    ],
    "multiple": [
        "Je détecte {count} {label}s aux positions : {bboxes}",
        "Il y a {count} {label}s visibles : {bboxes}",
        "Je vois {count} {label}s : {bboxes}",
        "Voici les {count} {label}s : {bboxes}",
    ]
}

def generate_detection_conversation(image_name, annotations):
    """Type 1 : Détection exhaustive de tous les composants"""
    components = []
    bboxes = []
    
    for ann in annotations:
        label = ann['label']
        bbox = normalize_bbox(ann['bbox'], ann['image_width'], ann['image_height'])
        components.append(f"{label} à {bbox_to_str(bbox)}")
        bboxes.append({
            'label': label,
            'bbox': bbox
        })
    
    question = random.choice(DETECTION_QUESTIONS)
    response = f"Je détecte : {', '.join(components)}"
    
    return {
        'image': image_name,
        'conversations': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': response}
        ],
        'bboxes': bboxes
    }

def generate_localization_conversation(image_name, annotations):
    """Type 2 : Localisation ciblée d'un composant spécifique"""
    # Choisit un composant aléatoire
    ann = random.choice(annotations)
    label = ann['label']
    bbox = normalize_bbox(ann['bbox'], ann['image_width'], ann['image_height'])
    
    # Question variée
    question = random.choice(LOCALIZATION_TEMPLATES.get(label, [f"Où est le {label} ?"]))
    
    # Réponse variée
    response = random.choice(RESPONSE_TEMPLATES["single"]).format(
        label=label,
        bbox=bbox_to_str(bbox)
    )
    
    return {
        'image': image_name,
        'conversations': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': response}
        ],
        'bboxes': [{
            'label': label,
            'bbox': bbox
        }]
    }

def generate_specific_component_conversation(image_name, annotations, target_label):
    """Type 4 : Question sur un composant spécifique (NOUVEAU - filtre les bboxes)"""
    # Filtre uniquement les annotations du composant ciblé
    filtered_anns = [ann for ann in annotations if ann['label'] == target_label]
    
    if not filtered_anns:
        return None
    
    # Normalise les bboxes
    bboxes = []
    bbox_strings = []
    for ann in filtered_anns:
        bbox = normalize_bbox(ann['bbox'], ann['image_width'], ann['image_height'])
        bboxes.append({
            'label': ann['label'],
            'bbox': bbox
        })
        bbox_strings.append(bbox_to_str(bbox))
    
    # Question variée
    question = random.choice(LOCALIZATION_TEMPLATES.get(target_label, [f"Où sont les {target_label}s ?"]))
    
    # Réponse selon le nombre
    if len(filtered_anns) == 1:
        response = random.choice(RESPONSE_TEMPLATES["single"]).format(
            label=target_label,
            bbox=bbox_strings[0]
        )
    else:
        response = random.choice(RESPONSE_TEMPLATES["multiple"]).format(
            count=len(filtered_anns),
            label=target_label,
            bboxes=", ".join(bbox_strings)
        )
    
    return {
        'image': image_name,
        'conversations': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': response}
        ],
        'bboxes': bboxes  # SEULEMENT les bboxes du composant demandé
    }

def generate_description_conversation(image_name, annotations):
    """Type 3 : Description générale de l'avion"""
    unique_labels = list(set(ann['label'] for ann in annotations))
    num_components = len(annotations)
    
    components_list = ', '.join(unique_labels[:-1]) + f" et {unique_labels[-1]}" if len(unique_labels) > 1 else unique_labels[0]
    
    bboxes = []
    for ann in annotations:
        bbox = normalize_bbox(ann['bbox'], ann['image_width'], ann['image_height'])
        bboxes.append({
            'label': ann['label'],
            'bbox': bbox
        })
    
    # Question variée
    question = random.choice(DESCRIPTION_QUESTIONS)
    
    # Réponse variée
    responses = [
        f"C'est un avion avec {num_components} composants visibles : {components_list}",
        f"Sur cette image, je vois un avion comportant : {components_list}",
        f"Cet appareil présente {num_components} éléments : {components_list}",
        f"L'avion comporte les composants suivants : {components_list}",
        f"Je peux identifier {num_components} parties : {components_list}",
    ]
    response = random.choice(responses)
    
    return {
        'image': image_name,
        'conversations': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': response}
        ],
        'bboxes': bboxes
    }

def verify_image_exists(image_name, images_dir):
    """Vérifie que l'image existe"""
    img_path = Path(images_dir) / image_name
    if not img_path.exists():
        return False
    
    try:
        img = Image.open(img_path)
        img.verify()
        return True
    except:
        return False

def create_dataset(annotations_by_image, images_dir):
    """Crée le dataset avec plusieurs types de conversations (décuplé)"""
    dataset = []
    
    print("🔄 Génération des conversations (mode décuplé)...")
    
    for image_name, annotations in annotations_by_image.items():
        # Vérifie que l'image existe
        if not verify_image_exists(image_name, images_dir):
            print(f"⚠️  Image non trouvée : {image_name}, skip")
            continue
        
        # Type 1 : Détection exhaustive (x2 variations)
        for _ in range(2):
            dataset.append(generate_detection_conversation(image_name, annotations))
        
        # Type 2 : Localisation aléatoire (x3 variations)
        for _ in range(3):
            dataset.append(generate_localization_conversation(image_name, annotations))
        
        # Type 3 : Description générale (x2 variations)
        for _ in range(2):
            dataset.append(generate_description_conversation(image_name, annotations))
        
        # Type 4 : Questions ciblées par composant (NOUVEAU - décuple le dataset)
        # Pour chaque type de composant présent dans l'image
        unique_labels = list(set(ann['label'] for ann in annotations))
        for label in unique_labels:
            # Génère 2 questions par composant présent
            for _ in range(2):
                conv = generate_specific_component_conversation(image_name, annotations, label)
                if conv:
                    dataset.append(conv)
    
    print(f"✅ {len(dataset)} conversations générées (dataset décuplé)")
    return dataset

def split_dataset(dataset, train_ratio=0.8):
    """Split train/val"""
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)
    
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    print(f"📊 Split : {len(train_data)} train, {len(val_data)} val")
    return train_data, val_data

def save_dataset(train_data, val_data, output_dir):
    """Sauvegarde les datasets"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / "train.json"
    val_path = output_path / "val.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Dataset sauvegardé dans {output_dir}")
    print(f"   - train.json : {len(train_data)} samples")
    print(f"   - val.json : {len(val_data)} samples")

def main():
    print("="*60)
    print("🚀 PRÉPARATION DU DATASET POUR OVIS DETECTION")
    print("="*60)
    
    # Charge les annotations
    annotations_by_image = load_and_group_annotations(CSV_PATH)
    
    # Crée le dataset
    dataset = create_dataset(annotations_by_image, IMAGES_DIR)
    
    # Split train/val
    train_data, val_data = split_dataset(dataset, TRAIN_RATIO)
    
    # Sauvegarde
    save_dataset(train_data, val_data, OUTPUT_DIR)
    
    # Statistiques
    print("\n📈 STATISTIQUES")
    print(f"   - Images uniques : {len(annotations_by_image)}")
    print(f"   - Total conversations : {len(dataset)}")
    print(f"   - Classes : {len(CLASSES)}")
    
    print("\n✅ DATASET PRÊT !")

if __name__ == "__main__":
    main()
