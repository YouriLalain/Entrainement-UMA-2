#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèle Qwen2.5-VL avec tête de détection
Architecture : Vision Encoder (frozen) → Detection Head → ROI Align → LLM (frozen)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision.ops import roi_align

class DetectionHead(nn.Module):
    """Tête de détection pour prédire les bounding boxes"""
    
    def __init__(self, in_channels=768, num_classes=12, hidden_dim=256):
        super().__init__()
        
        # Convolutions pour extraire les features spatiales
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # Heads de prédiction
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.bbox_head = nn.Conv2d(hidden_dim, 4, kernel_size=1)  # x, y, w, h
        
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] features du vision encoder
        Returns:
            cls_scores: [B, num_classes, H, W]
            bbox_preds: [B, 4, H, W]
        """
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        cls_scores = self.cls_head(x)
        bbox_preds = torch.sigmoid(self.bbox_head(x))  # Normalise en [0, 1]
        
        return cls_scores, bbox_preds


class ROIProjector(nn.Module):
    """Projette les features ROI vers l'espace du LLM"""
    
    def __init__(self, roi_size=7, in_channels=256, llm_dim=2048, hidden_dim=512):
        super().__init__()
        
        self.roi_size = roi_size
        self.pool_size = roi_size * roi_size * in_channels
        
        # MLP pour projeter vers la dimension du LLM
        self.projector = nn.Sequential(
            nn.Linear(self.pool_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )
        
    def forward(self, features, boxes):
        """
        Args:
            features: [B, C, H, W]
            boxes: List[Tensor] de shape [N, 4] pour chaque batch
        Returns:
            object_tokens: [B, num_objects, llm_dim]
        """
        if len(boxes) == 0:
            return None
        
        # ROI Align
        rois = roi_align(
            features,
            boxes,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0
        )  # [total_boxes, C, roi_size, roi_size]
        
        # Flatten et projette
        rois_flat = rois.flatten(1)  # [total_boxes, C*roi_size*roi_size]
        object_tokens = self.projector(rois_flat)  # [total_boxes, llm_dim]
        
        return object_tokens


class Qwen25VLDetectionModel(nn.Module):
    """Modèle Qwen2.5-VL avec détection"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config['num_classes']
        print("🔄 Chargement du modèle InternVL2-2B...")
        # Charge le tokenizer
        print("🔄 Chargement du tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True, use_fast=False)
        # Charge le modèle InternVL2-2B
        print("🔄 Chargement du modèle...")
        self.internvl_model = AutoModel.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        ).eval()
        # FREEZE tout le modèle InternVL2-2B
        print("❄️  Freeze du modèle InternVL2-2B...")
        for param in self.internvl_model.parameters():
            param.requires_grad = False
        
        # Détection dynamique de la dimension du vision encoder
        print("📐 Détection dynamique de la dimension du vision encoder...")
        # Crée dummy image avec le bon dtype et device
        dummy_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        dummy_device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy_img = torch.zeros(1, 3, 448, 448, dtype=dummy_dtype, device=dummy_device)
        with torch.no_grad():
                # InternVL2-2B attend [B, 3, 448, 448]
                vision_output = self.internvl_model.vision_model(dummy_img)
                features = vision_output.last_hidden_state if hasattr(vision_output, "last_hidden_state") else vision_output
                if len(features.shape) == 4:
                    vision_hidden_size = features.shape[1]
                elif len(features.shape) == 3:
                    vision_hidden_size = features.shape[2]
                else:
                    vision_hidden_size = 1536
        # LLM dim
        if hasattr(self.internvl_model.config, 'hidden_size'):
            llm_hidden_size = self.internvl_model.config.hidden_size
        elif hasattr(self.internvl_model.config, 'text_config'):
            llm_hidden_size = self.internvl_model.config.text_config.hidden_size
        else:
            llm_hidden_size = 2048
        print(f"📐 Dimensions détectées : vision={vision_hidden_size}, llm={llm_hidden_size}")
        # Modules entraînables
        print("🔧 Ajout de la tête de détection (trainable)...")
        self.detection_head = DetectionHead(
            in_channels=vision_hidden_size,
            num_classes=self.num_classes,
            hidden_dim=config['detection_hidden_dim']
        )
        
        print("🔧 Ajout du ROI projector (trainable)...")
        self.roi_projector = ROIProjector(
            roi_size=config['roi_size'],
            in_channels=config['detection_hidden_dim'],
            llm_dim=llm_hidden_size
        )
        
        # Sélection automatique du device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = torch.device(device)
        print(f"✅ Device sélectionné automatiquement : {self.device}")
        
        # Déplace les modules trainables sur le device
        print(f"🔄 Déplacement des modules trainables vers {self.device}...")
        self.detection_head = self.detection_head.to(self.device)
        self.roi_projector = self.roi_projector.to(self.device)
        
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Affiche le nombre de paramètres trainables"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📊 PARAMÈTRES:")
        print(f"   - Total: {total_params:,}")
        print(f"   - Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"   - Frozen: {total_params - trainable_params:,}\n")
    
    def extract_vision_features(self, images):
        """Extrait les features du vision encoder InternVL2-2B"""
        with torch.no_grad():
            try:
                # images: [B, 3, 448, 448]
                # Convertir au bon dtype si nécessaire (float16 sur CUDA)
                if torch.cuda.is_available() and images.dtype != torch.float16:
                    images = images.half()
                vision_output = self.internvl_model.vision_model(images)
                features = vision_output.last_hidden_state if hasattr(vision_output, "last_hidden_state") else vision_output
            except Exception as e:
                print(f"⚠️  Erreur extraction vision features: {e}")
                B = images.shape[0]
                C = self.detection_head.conv1.in_channels
                features = torch.randn(B, C, 28, 28).to(images.device)
            # Reshape si nécessaire
            if len(features.shape) == 3:
                B, num_patches, C = features.shape
                H = W = int(num_patches ** 0.5)
                if H * W == num_patches:
                    features = features.transpose(1, 2).reshape(B, C, H, W)
                else:
                    H = 28
                    W = (num_patches + H - 1) // H
                    if H * W > num_patches:
                        pad_size = H * W - num_patches
                        features = torch.cat([features, torch.zeros(B, pad_size, C).to(features.device)], dim=1)
                    features = features[:, :H*W, :].transpose(1, 2).reshape(B, C, H, W)
        return features
    
    def forward(self, images, text_inputs=None, target_boxes=None):
        """
        Forward pass
        
        Args:
            images: Tensor [B, C, H, W] ou output du processor
            text_inputs: Dict avec input_ids, attention_mask (optionnel)
            target_boxes: List[Tensor] ground truth boxes (optionnel)
            
        Returns:
            dict avec cls_scores, bbox_preds, text_output (si text_inputs fourni)
        """
        # Extrait les features visuelles (frozen)
        vision_features = self.extract_vision_features(images)
        
        # Passe dans la tête de détection (trainable)
        cls_scores, bbox_preds = self.detection_head(vision_features)
        
        outputs = {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds
        }
        
        # Si on a du texte, génère aussi la sortie textuelle
        if text_inputs is not None:
            # Projette les ROIs si on a des boxes
            object_tokens = None
            if target_boxes is not None and len(target_boxes) > 0:
                object_tokens = self.roi_projector(vision_features, target_boxes)
            
            # Forward LLM (frozen) - simplifié pour l'exemple
            with torch.no_grad():
                # Dans la vraie implémentation, tu injecterais object_tokens
                # dans le forward du LLM OVIS
                pass
            
            outputs['object_tokens'] = object_tokens
        
        return outputs
    
    def to(self, device):
        """Override to pour gérer MPS correctement"""
        self.device = device
        return super().to(device)


def load_model(config):
    """Charge le modèle"""
    model = Qwen25VLDetectionModel(config)
    # Le modèle est déjà sur le bon device via device_map
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Charge un checkpoint"""
    print(f"📂 Chargement du checkpoint : {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Charge seulement les weights trainables
    model.detection_head.load_state_dict(checkpoint['detection_head'])
    model.roi_projector.load_state_dict(checkpoint['roi_projector'])
    
    print("✅ Checkpoint chargé")
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))
