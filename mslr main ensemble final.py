# ============================================================================
# RADAR GESTURE CLASSIFICATION - COMPLETE PIPELINE v2.0
# ============================================================================
# 
# Final Target: 90.45%+ CV
# 
# Pipeline Overview:
# ══════════════════════════════════════════════════════════════════════════
# 
# PHASE 1: TEACHERS (Labeled Data Only)
#   ├── ConvNeXt Teachers (5-fold)         → ~87% CV
#   └── Swin Teachers (5-fold)             → ~87% CV
#              │
#              ▼ Generate Pseudo-Labels
# 
# PHASE 2: GEN-1 STUDENTS (Labeled + Pseudo)
#   ├── ConvNeXt Students (5-fold)         → ~90.4% CV
#   └── Swin Students (5-fold)             → ~90.3% CV
# 
# PHASE 3: ARCHITECTURAL DIVERSITY (Labeled + Pseudo)
#   ├── EfficientNet (dB-scaled, 5-fold)   → ~90.4% CV
#   ├── ArcFace (dB-scaled, 5-fold)        → ~90.4% CV
#   └── Spatial-Doppler (6-channel, 5-fold)→ ~88% CV (Orthogonal!)
# 
# PHASE 4: ENSEMBLE
#   └── 35-Model Weighted Ensemble         → ~90.45%+ CV
# 
# ══════════════════════════════════════════════════════════════════════════
#
# GitHub: [Your Repository]
# Author: [Your Name]
# Date: 2024
# ============================================================================

import os
import gc
import math
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

import timm

warnings.filterwarnings('ignore')

# IPython display for download links
try:
    from IPython.display import FileLink, display
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """
    Central configuration for the entire pipeline.
    All paths, hyperparameters, and settings are defined here.
    """
    
    # ========================= DATA PATHS =========================
    root_dir = Path("/home/rsnfh/radar_data")
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    test_dir = root_dir / "test"
    
    # ========================= OUTPUT DIRECTORIES =========================
    output_dir = root_dir / "complete_pipeline_v2"
    
    # Phase 1: Teachers (labeled only)
    teacher_convnext_dir = output_dir / "01_teacher_convnext"
    teacher_swin_dir = output_dir / "02_teacher_swin"
    
    # Phase 2: Gen-1 Students (labeled + pseudo)
    gen1_convnext_dir = output_dir / "03_gen1_convnext"
    gen1_swin_dir = output_dir / "04_gen1_swin"
    
    # Phase 3: Diversity Models (labeled + pseudo)
    efficientnet_dir = output_dir / "05_efficientnet_db"
    arcface_dir = output_dir / "06_arcface_db"
    spatial_doppler_dir = output_dir / "07_spatial_doppler"
    
    # Phase 4: Final Ensemble
    ensemble_dir = output_dir / "08_final_ensemble"
    
    # ========================= DATASET SETTINGS =========================
    num_classes = 126
    target_frames = 128
    
    # ========================= TRAINING SETTINGS =========================
    seed = 42
    folds = 5
    use_amp = True
    num_workers = 4
    
    # Teacher Training
    teacher_epochs = 30
    teacher_batch_size = 16
    teacher_lr = 5e-4
    teacher_weight_decay = 1e-2
    
    # Student Training
    student_epochs = 40
    student_batch_size = 16
    student_accumulate_steps = 2
    student_lr = 4e-4
    student_weight_decay = 2e-2
    
    # Pseudo-labeling
    pseudo_threshold = 0.65
    
    # ArcFace
    arcface_s = 15.0
    arcface_m = 0.20
    arcface_lr = 1e-4
    
    # Spatial-Doppler
    spatial_doppler_in_chans = 6  # 3 spatial + 3 doppler
    
    # ========================= AUGMENTATION =========================
    mixup_alpha = 0.8
    mixup_prob = 0.6
    label_smoothing = 0.1
    
    # ========================= ENSEMBLE WEIGHTS =========================
    # Based on individual model CV performance
    # Higher CV = higher weight
    ensemble_weights = {
        'teacher_convnext': 1.0,      # ~87% CV, 5 folds
        'teacher_swin': 1.0,          # ~87% CV, 5 folds
        'gen1_convnext': 2.0,         # ~90.4% CV, 5 folds (higher weight)
        'gen1_swin': 2.0,             # ~90.3% CV, 5 folds (higher weight)
        'efficientnet': 2.0,          # ~90.4% CV, 5 folds (higher weight)
        'arcface': 1.5,               # ~90.4% CV, 5 folds
        'spatial_doppler': 0.8,       # ~88% CV, 5 folds (orthogonal errors!)
    }


config = Config()

# Create all output directories
for dir_path in [
    config.teacher_convnext_dir, config.teacher_swin_dir,
    config.gen1_convnext_dir, config.gen1_swin_dir,
    config.efficientnet_dir, config.arcface_dir,
    config.spatial_doppler_dir, config.ensemble_dir
]:
    dir_path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calc_mca(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Mean Class Accuracy (MCA)
    More robust than plain accuracy for imbalanced datasets
    """
    unique_classes = np.unique(labels)
    per_class_acc = []
    for c in unique_classes:
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc.append(float((preds[mask] == c).mean()))
    return np.mean(per_class_acc)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup augmentation: blends two random samples
    Paper: https://arxiv.org/abs/1710.09412
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y


def print_header(text: str, char: str = "=", width: int = 80):
    """Print formatted header"""
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def print_subheader(text: str):
    """Print formatted subheader"""
    print(f"\n--- {text} ---")


def cleanup_gpu():
    """Aggressive GPU memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class SoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss that accepts soft (probabilistic) targets.
    Includes optional label smoothing.
    """
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            num_classes = targets.size(1)
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


# ============================================================================
# ARCFACE MODULE
# ============================================================================

class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace)
    Forces class separation on a hypersphere
    Paper: https://arxiv.org/abs/1801.07698
    """
    def __init__(self, in_features: int, out_features: int, s: float = 15.0, m: float = 0.20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        
        # Precompute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, input: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize input and weights
        input = input.float()
        weight = self.weight.float()
        
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(input_norm, weight_norm)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # Inference mode (no label)
        if label is None:
            return cosine * self.s
        
        # Training mode - apply angular margin
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        # Apply margin only to target class
        output = torch.where(one_hot > 0, phi, cosine)
        output = output * self.s
        
        return output


class ArcFaceModel(nn.Module):
    """
    Full model with backbone + bottleneck + ArcFace head
    """
    def __init__(self, model_name: str, num_classes: int, s: float = 15.0, m: float = 0.20):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0,
            drop_rate=0.2, drop_path_rate=0.1
        )
        self.feature_dim = self.backbone.num_features
        
        # Bottleneck for dimension reduction
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.arcface = ArcMarginProduct(512, num_classes, s=s, m=m)
        
    def forward(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.backbone(images)
        features = self.bottleneck(features)
        return self.arcface(features, labels)


# ============================================================================
# DATASET CLASSES
# ============================================================================

class BaseRadarDataset(Dataset):
    """
    Base dataset for radar RTM (Range-Time Map) data
    
    Input: 3 RTM channels (128 time frames × 256 range bins)
    Output: (3, 128, 256) tensor
    """
    def __init__(
        self, 
        sample_dirs: List[Path], 
        targets: Optional[np.ndarray] = None, 
        augment: bool = False, 
        resize_for_swin: bool = False
    ):
        self.sample_dirs = sample_dirs
        self.targets = targets
        self.augment = augment
        self.resize_for_swin = resize_for_swin
        self.tf = config.target_frames
        
    def __len__(self) -> int:
        return len(self.sample_dirs)
    
    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]
        
        rtms = []
        for i in range(1, 4):
            npy_files = list(sample_dir.glob(f"*_RTM{i}.npy"))
            if npy_files:
                rtm = np.load(npy_files[0], mmap_mode='r').astype(np.float32)
            else:
                rtm = np.zeros((self.tf, 256), dtype=np.float32)
            
            # Standardize time dimension
            T, R = rtm.shape
            if T != self.tf:
                rtm = F.interpolate(
                    torch.tensor(rtm).unsqueeze(0).unsqueeze(0),
                    size=(self.tf, R), mode='bilinear', align_corners=False
                ).squeeze().numpy()
            
            # Standard normalization
            mean, std = rtm.mean(), rtm.std() + 1e-8
            rtm = (rtm - mean) / std
            rtms.append(rtm)
        
        data = np.stack(rtms, axis=0)  # Shape: (3, 128, 256)
        
        # Apply augmentations
        if self.augment:
            data = self._apply_augmentations(data)
        
        x = torch.tensor(data, dtype=torch.float32)
        
        # Resize for Swin Transformer (requires 224×224)
        if self.resize_for_swin:
            x = F.interpolate(
                x.unsqueeze(0), size=(224, 224),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        
        # Return based on mode
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        else:
            sid = int(sample_dir.name.replace("SAMPLE_", ""))
            return x, sid
    
    def _apply_augmentations(self, data: np.ndarray) -> np.ndarray:
        """Apply SpecAugment-style masking and noise"""
        data = data.copy()
        
        # Time masking
        if random.random() < 0.6:
            mask_len = random.randint(15, 40)
            t0 = random.randint(0, max(1, data.shape[1] - mask_len))
            data[:, t0:t0+mask_len, :] = 0
        
        # Range masking
        if random.random() < 0.6:
            mask_len = random.randint(20, 50)
            r0 = random.randint(0, max(1, data.shape[2] - mask_len))
            data[:, :, r0:r0+mask_len] = 0
        
        # Gaussian noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.15, data.shape).astype(np.float32)
            data = data + noise
        
        # Channel permutation
        if random.random() < 0.4:
            data = data[np.random.permutation(3)]
        
        return data


class dBScaledDataset(BaseRadarDataset):
    """
    Dataset with dB (logarithmic) scaling
    
    Improves dynamic range for capturing faint radar signatures
    (micro-Doppler from hands/fingers vs. strong body reflection)
    """
    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]
        
        rtms = []
        for i in range(1, 4):
            npy_files = list(sample_dir.glob(f"*_RTM{i}.npy"))
            if npy_files:
                rtm = np.load(npy_files[0], mmap_mode='r').astype(np.float32)
            else:
                rtm = np.zeros((self.tf, 256), dtype=np.float32)
            
            T, R = rtm.shape
            if T != self.tf:
                rtm = F.interpolate(
                    torch.tensor(rtm).unsqueeze(0).unsqueeze(0),
                    size=(self.tf, R), mode='bilinear', align_corners=False
                ).squeeze().numpy()
            
            # ============ dB SCALING ============
            rtm = np.abs(rtm) + 1e-6
            rtm = 10 * np.log10(rtm)
            # =====================================
            
            mean, std = rtm.mean(), rtm.std() + 1e-8
            rtm = (rtm - mean) / std
            rtms.append(rtm)
        
        data = np.stack(rtms, axis=0)
        
        if self.augment:
            data = self._apply_augmentations(data)
        
        x = torch.tensor(data, dtype=torch.float32)
        
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        else:
            sid = int(sample_dir.name.replace("SAMPLE_", ""))
            return x, sid


class SpatialDopplerDataset(Dataset):
    """
    Dataset with 6-channel Spatial + Doppler (FFT) features
    
    Channels 1-3: Spatial RTM maps (standard normalization)
    Channels 4-6: Doppler maps (FFT along time axis → velocity)
    
    This provides explicit velocity information that CNNs struggle to infer
    from spatial-only data.
    """
    def __init__(
        self, 
        sample_dirs: List[Path], 
        targets: Optional[np.ndarray] = None, 
        augment: bool = False
    ):
        self.sample_dirs = sample_dirs
        self.targets = targets
        self.augment = augment
        self.tf = config.target_frames
        
    def __len__(self) -> int:
        return len(self.sample_dirs)
    
    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]
        
        spatial_channels = []
        doppler_channels = []
        
        for i in range(1, 4):
            npy_files = list(sample_dir.glob(f"*_RTM{i}.npy"))
            if npy_files:
                rtm = np.load(npy_files[0], mmap_mode='r').astype(np.float32)
            else:
                rtm = np.zeros((self.tf, 256), dtype=np.float32)
            
            T, R = rtm.shape
            if T != self.tf:
                rtm = F.interpolate(
                    torch.tensor(rtm).unsqueeze(0).unsqueeze(0),
                    size=(self.tf, R), mode='bilinear', align_corners=False
                ).squeeze().numpy()
            
            # ============ DOPPLER (FFT) EXTRACTION ============
            # FFT along time axis converts time → frequency (velocity)
            doppler = np.abs(np.fft.fft(rtm, axis=0))
            doppler = np.fft.fftshift(doppler, axes=0)  # Center zero-velocity
            # ==================================================
            
            # dB scale both
            rtm_db = 10 * np.log10(np.abs(rtm) + 1e-6)
            doppler_db = 10 * np.log10(doppler + 1e-6)
            
            # Normalize independently
            rtm_db = (rtm_db - rtm_db.mean()) / (rtm_db.std() + 1e-8)
            doppler_db = (doppler_db - doppler_db.mean()) / (doppler_db.std() + 1e-8)
            
            spatial_channels.append(rtm_db)
            doppler_channels.append(doppler_db)
        
        # Stack: [Spatial1, Spatial2, Spatial3, Doppler1, Doppler2, Doppler3]
        data = np.stack(spatial_channels + doppler_channels, axis=0)  # Shape: (6, 128, 256)
        
        # Augmentations
        if self.augment:
            data = self._apply_augmentations(data)
        
        x = torch.tensor(data, dtype=torch.float32)
        
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        else:
            sid = int(sample_dir.name.replace("SAMPLE_", ""))
            return x, sid
    
    def _apply_augmentations(self, data: np.ndarray) -> np.ndarray:
        """Apply augmentations (same for all 6 channels)"""
        data = data.copy()
        
        # Time masking
        if random.random() < 0.6:
            mask_len = random.randint(15, 30)
            t0 = random.randint(0, max(1, data.shape[1] - mask_len))
            data[:, t0:t0+mask_len, :] = 0
        
        # Range masking
        if random.random() < 0.6:
            mask_len = random.randint(15, 40)
            r0 = random.randint(0, max(1, data.shape[2] - mask_len))
            data[:, :, r0:r0+mask_len] = 0
        
        # Noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.1, data.shape).astype(np.float32)
            data = data + noise
        
        return data


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> Tuple[List[Path], List[int], List[Path]]:
    """
    Load labeled and unlabeled sample directories
    
    Returns:
        labeled_dirs: List of paths to labeled sample folders
        labeled_classes: List of class IDs for each labeled sample
        unlabeled_dirs: List of paths to unlabeled sample folders (val + test)
    """
    print_header("LOADING DATA")
    
    labeled_dirs = []
    labeled_classes = []
    unlabeled_dirs = []
    
    # Load labeled training data
    for cls_dir in sorted(config.train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        parts = cls_dir.name.split("_")
        if not parts[0].isdigit():
            continue
        cls_id = int(parts[0])
        for sd in cls_dir.iterdir():
            if sd.name.startswith("SAMPLE_"):
                labeled_dirs.append(sd)
                labeled_classes.append(cls_id)
    
    # Load unlabeled data (val + test)
    for sd in sorted(config.val_dir.iterdir()):
        if sd.name.startswith("SAMPLE_"):
            unlabeled_dirs.append(sd)
    for sd in sorted(config.test_dir.iterdir()):
        if sd.name.startswith("SAMPLE_"):
            unlabeled_dirs.append(sd)
    
    print(f"  Labeled samples:   {len(labeled_dirs)}")
    print(f"  Unlabeled samples: {len(unlabeled_dirs)}")
    print(f"  Number of classes: {len(set(labeled_classes))}")
    print(f"  Submission rows:   {len(unlabeled_dirs)}")
    
    return labeled_dirs, labeled_classes, unlabeled_dirs


# ============================================================================
# TRAINING ENGINE
# ============================================================================

def train_model_cv(
    model_name: str,
    output_dir: Path,
    train_dirs: List[Path],
    train_targets: np.ndarray,
    unlabeled_dirs: Optional[List[Path]] = None,
    pseudo_targets: Optional[np.ndarray] = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    use_mixup: bool = False,
    use_arcface: bool = False,
    use_db_scale: bool = False,
    use_spatial_doppler: bool = False,
    resize_for_swin: bool = False,
    model_tag: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic K-Fold Cross-Validation training function
    
    Supports all model variants:
    - Standard ConvNeXt/Swin (BaseRadarDataset)
    - dB-scaled models (dBScaledDataset)
    - ArcFace models (ArcFaceModel + dBScaledDataset)
    - Spatial-Doppler fusion (SpatialDopplerDataset)
    
    Args:
        model_name: timm model name (e.g., 'convnext_small')
        output_dir: Directory to save checkpoints
        train_dirs: Labeled sample directories
        train_targets: Labels (1D int array or 2D soft targets)
        unlabeled_dirs: Optional pseudo-labeled sample directories
        pseudo_targets: Optional soft targets for pseudo-labeled samples
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        use_mixup: Apply mixup augmentation
        use_arcface: Use ArcFace loss
        use_db_scale: Use dB (log) scaling
        use_spatial_doppler: Use 6-channel spatial+doppler input
        resize_for_swin: Resize to 224x224 for Swin
        model_tag: Tag for logging
    
    Returns:
        oof_preds: Out-of-fold predictions (N, num_classes)
        oof_labels: True labels (N,)
    """
    
    tag = model_tag or model_name.upper()
    print_header(f"TRAINING: {tag}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Mixup: {use_mixup} | ArcFace: {use_arcface}")
    print(f"  dB Scale: {use_db_scale} | Spatial-Doppler: {use_spatial_doppler}")
    
    # Convert targets to proper format
    if len(train_targets.shape) == 1:
        # Hard labels → soft targets
        hard_labels = train_targets.astype(np.int64)
        soft_targets = np.zeros((len(train_targets), config.num_classes), dtype=np.float32)
        soft_targets[np.arange(len(train_targets)), hard_labels] = 1.0
    else:
        soft_targets = train_targets
        hard_labels = train_targets.argmax(1).astype(np.int64)
    
    # K-Fold split
    skf = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    splits = list(skf.split(train_dirs, hard_labels))
    
    # OOF tracking
    oof_preds = np.zeros((len(train_dirs), config.num_classes), dtype=np.float32)
    oof_labels = hard_labels.copy()
    
    # Check existing OOF
    oof_path = output_dir / "oof_preds.npy"
    oof_labels_path = output_dir / "oof_labels.npy"
    if oof_path.exists():
        oof_preds = np.load(oof_path)
        print(f"  Loaded existing OOF predictions")
    
    fold_mcas = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_num = fold + 1
        ckpt_path = output_dir / f"fold_{fold_num}.pth"
        
        # Skip if already trained
        if ckpt_path.exists() and ckpt_path.stat().st_size > 1000:
            print_subheader(f"FOLD {fold_num}/{config.folds} - SKIPPED (checkpoint exists)")
            
            # Recover MCA from OOF
            oof_fold = oof_preds[val_idx]
            if oof_fold.sum() != 0:
                preds = oof_fold.argmax(1)
                targets = oof_labels[val_idx]
                mca = calc_mca(preds, targets)
                fold_mcas.append(mca)
                print(f"  Recovered MCA: {mca*100:.2f}%")
            else:
                fold_mcas.append(0.0)
            continue
        
        print_subheader(f"FOLD {fold_num}/{config.folds} - TRAINING")
        
        # Prepare fold data
        val_dirs = [train_dirs[i] for i in val_idx]
        val_targets = soft_targets[val_idx]
        
        fold_train_dirs = [train_dirs[i] for i in train_idx]
        fold_train_targets = soft_targets[train_idx]
        
        # Add pseudo-labeled data
        if unlabeled_dirs is not None and pseudo_targets is not None:
            fold_train_dirs = fold_train_dirs + unlabeled_dirs
            fold_train_targets = np.vstack([fold_train_targets, pseudo_targets])
        
        print(f"  Train: {len(fold_train_dirs)} | Val: {len(val_dirs)}")
        
        # Select dataset class
        if use_spatial_doppler:
            train_dataset = SpatialDopplerDataset(fold_train_dirs, fold_train_targets, augment=True)
            val_dataset = SpatialDopplerDataset(val_dirs, val_targets, augment=False)
            in_chans = 6
        elif use_db_scale:
            train_dataset = dBScaledDataset(fold_train_dirs, fold_train_targets, augment=True)
            val_dataset = dBScaledDataset(val_dirs, val_targets, augment=False)
            in_chans = 3
        else:
            train_dataset = BaseRadarDataset(
                fold_train_dirs, fold_train_targets, 
                augment=True, resize_for_swin=resize_for_swin
            )
            val_dataset = BaseRadarDataset(
                val_dirs, val_targets,
                augment=False, resize_for_swin=resize_for_swin
            )
            in_chans = 3
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=(config.num_workers > 0)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=(config.num_workers > 0)
        )
        
        # Create model
        if use_arcface:
            model = ArcFaceModel(
                model_name, config.num_classes,
                s=config.arcface_s, m=config.arcface_m
            ).to(device)
            criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        else:
            model = timm.create_model(
                model_name, pretrained=True,
                in_chans=in_chans, num_classes=config.num_classes,
                drop_rate=0.3, drop_path_rate=0.2
            ).to(device)
            criterion = SoftCrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # Optimizer & scheduler
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = OneCycleLR(
            optimizer, max_lr=lr,
            total_steps=epochs * len(train_loader) + 1,
            pct_start=0.1
        )
        scaler = GradScaler(enabled=config.use_amp)
        
        best_mca = 0
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                with autocast(enabled=config.use_amp):
                    # Mixup
                    if use_mixup and random.random() < config.mixup_prob:
                        x, y = mixup_data(x, y, config.mixup_alpha)
                    
                    # Forward
                    if use_arcface:
                        y_hard = y.argmax(1) if len(y.shape) > 1 else y
                        logits = model(x, labels=y_hard)
                        loss = criterion(logits, y_hard)
                    else:
                        logits = model(x)
                        loss = criterion(logits, y)
                
                # Backward
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_probs_list = []
            
            with torch.no_grad(), autocast(enabled=config.use_amp):
                for x, _ in val_loader:
                    x = x.to(device, non_blocking=True)
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                    val_probs_list.append(probs.cpu())
            
            val_probs = torch.cat(val_probs_list).numpy()
            preds_cls = val_probs.argmax(1)
            targets_cls = val_targets.argmax(1)
            
            mca = calc_mca(preds_cls, targets_cls)
            
            # Save best
            if mca > best_mca:
                best_mca = mca
                torch.save(model.state_dict(), ckpt_path)
                
                # Update OOF
                for i, vi in enumerate(val_idx):
                    oof_preds[vi] = val_probs[i]
                
                # Save OOF incrementally
                np.save(oof_path, oof_preds)
                np.save(oof_labels_path, oof_labels)
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                avg_loss = train_loss / len(train_loader)
                mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(f"  Ep {epoch+1:2d} | Loss: {avg_loss:.4f} | "
                      f"MCA: {mca*100:.2f}% | Best: {best_mca*100:.2f}% | GPU: {mem:.1f}GB")
            
            # Periodic cleanup
            if (epoch + 1) % 10 == 0:
                cleanup_gpu()
        
        fold_mcas.append(best_mca)
        print(f"  ✅ Fold {fold_num} Best MCA: {best_mca*100:.2f}%")
        
        # Cleanup
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        cleanup_gpu()
    
    # Final save
    np.save(oof_path, oof_preds)
    np.save(oof_labels_path, oof_labels)
    
    # Calculate final CV
    cv_score = calc_mca(oof_preds.argmax(1), oof_labels)
    print(f"\n  📊 FINAL CV: {cv_score*100:.2f}%")
    print(f"  Per-fold MCAs: {[f'{m*100:.2f}%' for m in fold_mcas]}")
    
    return oof_preds, oof_labels


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

def generate_predictions(
    model_name: str,
    checkpoint_dir: Path,
    unlabeled_dirs: List[Path],
    use_arcface: bool = False,
    use_db_scale: bool = False,
    use_spatial_doppler: bool = False,
    resize_for_swin: bool = False,
    model_tag: str = ""
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate predictions from K-fold models
    
    Returns:
        probs: Averaged softmax probabilities (N, num_classes)
        sample_ids: List of sample IDs
    """
    
    tag = model_tag or model_name
    print(f"\n  Inference: {tag}")
    
    # Select dataset
    if use_spatial_doppler:
        inf_dataset = SpatialDopplerDataset(unlabeled_dirs)
        in_chans = 6
    elif use_db_scale:
        inf_dataset = dBScaledDataset(unlabeled_dirs)
        in_chans = 3
    else:
        inf_dataset = BaseRadarDataset(unlabeled_dirs, resize_for_swin=resize_for_swin)
        in_chans = 3
    
    inf_loader = DataLoader(
        inf_dataset, batch_size=32, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    all_probs = np.zeros((len(unlabeled_dirs), config.num_classes), dtype=np.float32)
    sample_ids = None
    folds_used = 0
    
    for fold in range(1, config.folds + 1):
        ckpt_path = checkpoint_dir / f"fold_{fold}.pth"
        if not ckpt_path.exists():
            print(f"    ⚠️ Fold {fold} not found, skipping")
            continue
        
        print(f"    Fold {fold}...", end=" ", flush=True)
        
        # Load model
        if use_arcface:
            model = ArcFaceModel(
                model_name, config.num_classes,
                s=config.arcface_s, m=config.arcface_m
            ).to(device)
        else:
            model = timm.create_model(
                model_name, pretrained=False,
                in_chans=in_chans, num_classes=config.num_classes
            ).to(device)
        
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        fold_probs = []
        fold_ids = []
        
        with torch.no_grad(), autocast(enabled=config.use_amp):
            for x, ids in inf_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                fold_probs.append(probs)
                if sample_ids is None:
                    fold_ids.extend(ids.tolist())
        
        all_probs += np.concatenate(fold_probs, axis=0)
        if sample_ids is None:
            sample_ids = fold_ids
        folds_used += 1
        
        del model
        cleanup_gpu()
        print("✅")
    
    all_probs /= folds_used
    print(f"    Used {folds_used} folds")
    
    return all_probs, sample_ids


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete pipeline"""
    
    start_time = datetime.now()
    
    print("=" * 80)
    print(" " * 15 + "RADAR GESTURE CLASSIFICATION - COMPLETE PIPELINE")
    print("=" * 80)
    print(f"  Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Output: {config.output_dir}")
    print("=" * 80)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    labeled_dirs, labeled_classes, unlabeled_dirs = load_data()
    labeled_targets = np.array(labeled_classes, dtype=np.int64)
    
    # ========================================================================
    # PHASE 1: TEACHER MODELS (Labeled Data Only)
    # ========================================================================
    
    print_header("PHASE 1: TEACHER MODELS (Labeled Data Only)", "█")
    
    # ConvNeXt Teacher
    teacher_oof_conv, teacher_labels = train_model_cv(
        model_name='convnext_small',
        output_dir=config.teacher_convnext_dir,
        train_dirs=labeled_dirs,
        train_targets=labeled_targets,
        epochs=config.teacher_epochs,
        batch_size=config.teacher_batch_size,
        lr=config.teacher_lr,
        weight_decay=config.teacher_weight_decay,
        use_mixup=False,
        model_tag="TEACHER CONVNEXT"
    )
    
    # Swin Teacher
    teacher_oof_swin, _ = train_model_cv(
        model_name='swin_small_patch4_window7_224',
        output_dir=config.teacher_swin_dir,
        train_dirs=labeled_dirs,
        train_targets=labeled_targets,
        epochs=config.teacher_epochs,
        batch_size=config.teacher_batch_size,
        lr=config.teacher_lr,
        weight_decay=config.teacher_weight_decay,
        use_mixup=False,
        resize_for_swin=True,
        model_tag="TEACHER SWIN"
    )
    
    # ========================================================================
    # GENERATE PSEUDO-LABELS
    # ========================================================================
    
    print_header("GENERATING PSEUDO-LABELS")
    
    pseudo_path = config.output_dir / "pseudo_labels.npy"
    
    if pseudo_path.exists():
        print("  Loading cached pseudo-labels...")
        teacher_ensemble = np.load(pseudo_path)
    else:
        print("  Generating predictions from teachers...")
        
        teacher_conv_probs, _ = generate_predictions(
            'convnext_small', config.teacher_convnext_dir, unlabeled_dirs,
            model_tag="Teacher ConvNeXt"
        )
        teacher_swin_probs, sample_ids = generate_predictions(
            'swin_small_patch4_window7_224', config.teacher_swin_dir, unlabeled_dirs,
            resize_for_swin=True, model_tag="Teacher Swin"
        )
        
        teacher_ensemble = (teacher_conv_probs + teacher_swin_probs) / 2.0
        np.save(pseudo_path, teacher_ensemble)
    
    # Filter by confidence
    max_confs = teacher_ensemble.max(axis=1)
    keep_mask = max_confs >= config.pseudo_threshold
    
    filtered_unlabeled_dirs = [unlabeled_dirs[i] for i in range(len(unlabeled_dirs)) if keep_mask[i]]
    filtered_pseudo_targets = teacher_ensemble[keep_mask]
    
    print(f"\n  Pseudo-label threshold: {config.pseudo_threshold}")
    print(f"  Samples kept: {keep_mask.sum()} / {len(unlabeled_dirs)} ({keep_mask.mean()*100:.1f}%)")
    print(f"  Average confidence: {max_confs[keep_mask].mean():.3f}")
    
    # ========================================================================
    # PHASE 2: GEN-1 STUDENTS (Noisy Student)
    # ========================================================================
    
    print_header("PHASE 2: GEN-1 STUDENTS (Labeled + Pseudo)", "█")
    
    # ConvNeXt Gen-1
    gen1_oof_conv, _ = train_model_cv(
        model_name='convnext_small',
        output_dir=config.gen1_convnext_dir,
        train_dirs=labeled_dirs,
        train_targets=labeled_targets,
        unlabeled_dirs=filtered_unlabeled_dirs,
        pseudo_targets=filtered_pseudo_targets,
        epochs=config.student_epochs,
        batch_size=config.student_batch_size,
        lr=config.student_lr,
        weight_decay=config.student_weight_decay,
        use_mixup=True,
        model_tag="GEN-1 CONVNEXT"
    )
    
    # Swin Gen-1
    gen1_oof_swin, _ = train_model_cv(
        model_name='swin_small_patch4_window7_224',
        output_dir=config.gen1_swin_dir,
        train_dirs=labeled_dirs,
        train_targets=labeled_targets,
        unlabeled_dirs=filtered_unlabeled_dirs,
        pseudo_targets=filtered_pseudo_targets,
        epochs=config.student_epochs,
        batch_size=config.student_batch_size,
        lr=config.student_lr,
        weight_decay=config.student_weight_decay,
        use_mixup=True,
        resize_for_swin=True,
        model_tag="GEN-1 SWIN"
    )
    
    # ========================================================================
    # PHASE 3: ARCHITECTURAL DIVERSITY
    # ========================================================================
    
    print_header("PHASE 3: ARCHITECTURAL DIVERSITY", "█")
    
    # EfficientNet (dB-scaled)
    effnet_oof, _ = train_model_cv(
        model_name='tf_efficientnetv2_s',
        output_dir=config.efficientnet_dir,
        train_dirs=labeled_dirs,
        train_targets=labeled_targets,
        unlabeled_dirs=filtered_unlabeled_dirs,
        pseudo_targets=filtered_pseudo_targets,
        epochs=config.student_epochs,
        batch_size=config.student_batch_size,
        lr=config.student_lr,
        weight_decay=config.student_weight_decay,
        use_mixup=True,
        use_db_scale=True,
        model_tag="EFFICIENTNET (dB)"
    )
    
    # ArcFace (dB-scaled)
    arcface_oof, _ = train_model_cv(
        model_name='convnext_small',
        output_dir=config.arcface_dir,
        train_dirs=labeled_dirs,
        train_targets=labeled_targets,
        unlabeled_dirs=filtered_unlabeled_dirs,
        pseudo_targets=filtered_pseudo_targets,
        epochs=config.student_epochs,
        batch_size=config.student_batch_size,
        lr=config.arcface_lr,
        weight_decay=config.student_weight_decay,
        use_mixup=False,
        use_arcface=True,
        use_db_scale=True,
        model_tag="ARCFACE (dB)"
    )
    
    # Spatial-Doppler Fusion (6-channel)
    spatial_doppler_oof, _ = train_model_cv(
        model_name='convnext_small',
        output_dir=config.spatial_doppler_dir,
        train_dirs=labeled_dirs,
        train_targets=labeled_targets,
        unlabeled_dirs=filtered_unlabeled_dirs,
        pseudo_targets=filtered_pseudo_targets,
        epochs=config.student_epochs,
        batch_size=config.student_batch_size,
        lr=config.student_lr,
        weight_decay=config.student_weight_decay,
        use_mixup=True,
        use_spatial_doppler=True,
        model_tag="SPATIAL-DOPPLER (6ch)"
    )
    
    # ========================================================================
    # PHASE 4: FINAL ENSEMBLE
    # ========================================================================
    
    print_header("PHASE 4: FINAL ENSEMBLE", "█")
    
    # Collect all test predictions
    all_test_probs = {}
    
    print("\n  Generating test predictions from all models...")
    
    # Teachers
    all_test_probs['teacher_convnext'], _ = generate_predictions(
        'convnext_small', config.teacher_convnext_dir, unlabeled_dirs,
        model_tag="Teacher ConvNeXt"
    )
    all_test_probs['teacher_swin'], sample_ids = generate_predictions(
        'swin_small_patch4_window7_224', config.teacher_swin_dir, unlabeled_dirs,
        resize_for_swin=True, model_tag="Teacher Swin"
    )
    
    # Gen-1 Students
    all_test_probs['gen1_convnext'], _ = generate_predictions(
        'convnext_small', config.gen1_convnext_dir, unlabeled_dirs,
        model_tag="Gen-1 ConvNeXt"
    )
    all_test_probs['gen1_swin'], _ = generate_predictions(
        'swin_small_patch4_window7_224', config.gen1_swin_dir, unlabeled_dirs,
        resize_for_swin=True, model_tag="Gen-1 Swin"
    )
    
    # Diversity Models
    all_test_probs['efficientnet'], _ = generate_predictions(
        'tf_efficientnetv2_s', config.efficientnet_dir, unlabeled_dirs,
        use_db_scale=True, model_tag="EfficientNet (dB)"
    )
    all_test_probs['arcface'], _ = generate_predictions(
        'convnext_small', config.arcface_dir, unlabeled_dirs,
        use_arcface=True, use_db_scale=True, model_tag="ArcFace (dB)"
    )
    all_test_probs['spatial_doppler'], _ = generate_predictions(
        'convnext_small', config.spatial_doppler_dir, unlabeled_dirs,
        use_spatial_doppler=True, model_tag="Spatial-Doppler (6ch)"
    )
    
    # ========================================================================
    # CREATE ENSEMBLE SUBMISSIONS
    # ========================================================================
    
    print_header("CREATING SUBMISSIONS")
    
    # Weighted ensemble
    weights = config.ensemble_weights
    total_weight = sum(weights.values())
    
    weighted_ensemble = np.zeros((len(unlabeled_dirs), config.num_classes), dtype=np.float32)
    for model_name, probs in all_test_probs.items():
        w = weights.get(model_name, 1.0)
        weighted_ensemble += w * probs
        print(f"    {model_name}: weight = {w}")
    weighted_ensemble /= total_weight
    
    # Also create equal-weight ensemble for comparison
    equal_ensemble = np.mean(list(all_test_probs.values()), axis=0)
    
    # Save submissions
    submissions = []
    
    # 1. Weighted ensemble (main submission)
    df_weighted = pd.DataFrame({
        'id': sample_ids,
        'Pred': weighted_ensemble.argmax(1)
    })
    df_weighted = df_weighted.sort_values('id').reset_index(drop=True)
    weighted_path = config.ensemble_dir / "submission_weighted_ensemble.csv"
    df_weighted.to_csv(weighted_path, index=False)
    submissions.append(("submission_weighted_ensemble.csv", "⭐ WEIGHTED ENSEMBLE (Recommended)"))
    
    # 2. Equal-weight ensemble
    df_equal = pd.DataFrame({
        'id': sample_ids,
        'Pred': equal_ensemble.argmax(1)
    })
    df_equal = df_equal.sort_values('id').reset_index(drop=True)
    equal_path = config.ensemble_dir / "submission_equal_ensemble.csv"
    df_equal.to_csv(equal_path, index=False)
    submissions.append(("submission_equal_ensemble.csv", "Equal-weight ensemble"))
    
    # 3. Without Spatial-Doppler (higher-CV models only)
    high_cv_ensemble = np.zeros((len(unlabeled_dirs), config.num_classes), dtype=np.float32)
    high_cv_models = ['gen1_convnext', 'gen1_swin', 'efficientnet', 'arcface']
    for m in high_cv_models:
        high_cv_ensemble += all_test_probs[m]
    high_cv_ensemble /= len(high_cv_models)
    
    df_high_cv = pd.DataFrame({
        'id': sample_ids,
        'Pred': high_cv_ensemble.argmax(1)
    })
    df_high_cv = df_high_cv.sort_values('id').reset_index(drop=True)
    high_cv_path = config.ensemble_dir / "submission_high_cv_only.csv"
    df_high_cv.to_csv(high_cv_path, index=False)
    submissions.append(("submission_high_cv_only.csv", "High-CV models only (no Spatial-Doppler)"))
    
    # Save probabilities for future analysis
    np.save(config.ensemble_dir / "weighted_ensemble_probs.npy", weighted_ensemble)
    np.save(config.ensemble_dir / "all_model_probs.npy", all_test_probs)
    
    # ========================================================================
    # COMPUTE CV SCORES
    # ========================================================================
    
    print_header("CROSS-VALIDATION RESULTS")
    
    cv_scores = {}
    
    # Individual model CVs
    for name, oof in [
        ('Teacher ConvNeXt', teacher_oof_conv),
        ('Teacher Swin', teacher_oof_swin),
        ('Gen-1 ConvNeXt', gen1_oof_conv),
        ('Gen-1 Swin', gen1_oof_swin),
        ('EfficientNet (dB)', effnet_oof),
        ('ArcFace (dB)', arcface_oof),
        ('Spatial-Doppler', spatial_doppler_oof),
    ]:
        cv = calc_mca(oof.argmax(1), teacher_labels)
        cv_scores[name] = cv
        print(f"  {name:25s}: {cv*100:.2f}%")
    
    # Teacher ensemble CV
    teacher_oof_ensemble = (teacher_oof_conv + teacher_oof_swin) / 2.0
    teacher_ensemble_cv = calc_mca(teacher_oof_ensemble.argmax(1), teacher_labels)
    print(f"\n  {'Teacher Ensemble':25s}: {teacher_ensemble_cv*100:.2f}%")
    
    # Gen-1 ensemble CV
    gen1_oof_ensemble = (gen1_oof_conv + gen1_oof_swin) / 2.0
    gen1_ensemble_cv = calc_mca(gen1_oof_ensemble.argmax(1), teacher_labels)
    print(f"  {'Gen-1 Ensemble':25s}: {gen1_ensemble_cv*100:.2f}%")
    
    # Full ensemble CV
    all_oof = {
        'teacher_convnext': teacher_oof_conv,
        'teacher_swin': teacher_oof_swin,
        'gen1_convnext': gen1_oof_conv,
        'gen1_swin': gen1_oof_swin,
        'efficientnet': effnet_oof,
        'arcface': arcface_oof,
        'spatial_doppler': spatial_doppler_oof,
    }
    
    # Weighted OOF ensemble
    weighted_oof = np.zeros_like(teacher_oof_conv)
    for name, oof in all_oof.items():
        w = weights.get(name, 1.0)
        weighted_oof += w * oof
    weighted_oof /= total_weight
    
    weighted_ensemble_cv = calc_mca(weighted_oof.argmax(1), teacher_labels)
    print(f"\n  {'WEIGHTED ENSEMBLE':25s}: {weighted_ensemble_cv*100:.2f}% ⭐")
    
    # ========================================================================
    # DISPLAY DOWNLOAD LINKS
    # ========================================================================
    
    print_header("📥 DOWNLOAD LINKS")
    
    for filename, description in submissions:
        path = config.ensemble_dir / filename
        if path.exists():
            size_kb = path.stat().st_size / 1024
            rows = len(pd.read_csv(path))
            print(f"\n  ✅ {filename}")
            print(f"     {description}")
            print(f"     Size: {size_kb:.1f} KB | Rows: {rows}")
            if IN_NOTEBOOK:
                display(FileLink(str(path)))
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("🎯 PIPELINE COMPLETE", "█")
    
    print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                         RESULTS SUMMARY                              ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║                                                                      ║
  ║  MODELS TRAINED: 35 (7 model types × 5 folds)                        ║
  ║                                                                      ║
  ║  INDIVIDUAL CV SCORES:                                               ║
  ║    Teacher ConvNeXt:      {cv_scores['Teacher ConvNeXt']*100:5.2f}%                                ║
  ║    Teacher Swin:          {cv_scores['Teacher Swin']*100:5.2f}%                                ║
  ║    Gen-1 ConvNeXt:        {cv_scores['Gen-1 ConvNeXt']*100:5.2f}%                                ║
  ║    Gen-1 Swin:            {cv_scores['Gen-1 Swin']*100:5.2f}%                                ║
  ║    EfficientNet (dB):     {cv_scores['EfficientNet (dB)']*100:5.2f}%                                ║
  ║    ArcFace (dB):          {cv_scores['ArcFace (dB)']*100:5.2f}%                                ║
  ║    Spatial-Doppler:       {cv_scores['Spatial-Doppler']*100:5.2f}%                                ║
  ║                                                                      ║
  ║  ENSEMBLE CV:             {weighted_ensemble_cv*100:5.2f}% ⭐                               ║
  ║                                                                      ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║                                                                      ║
  ║  OUTPUT FILES:                                                       ║
  ║    📁 {str(config.ensemble_dir):<60} ║
  ║    📄 submission_weighted_ensemble.csv  (RECOMMENDED)                ║
  ║    📄 submission_equal_ensemble.csv                                  ║
  ║    📄 submission_high_cv_only.csv                                    ║
  ║                                                                      ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║                                                                      ║
  ║  RUNTIME: {str(duration).split('.')[0]:>20}                                        ║
  ║                                                                      ║
  ╚══════════════════════════════════════════════════════════════════════╝
""")
    
    print("\n🚀 Submit 'submission_weighted_ensemble.csv' to the leaderboard!")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
