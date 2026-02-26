
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

import matplotlib
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

# ⚠️ CHANGE THIS TO YOUR DATA PATH
DATA_ROOT = r'D:/Jafar/Ultrasound/BUSI/All/'

IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 1000
BASE_LR = 5e-5
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
NUM_WORKERS = 0  # Windows safe
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = r'D:/Jafar/Ultrasound/results/'

# ============================================================================
# SETUP
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print(f"\n{'='*80}")
print(f"BREAST ULTRASOUND SEGMENTATION")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
print(f"Data: {DATA_ROOT}")
print(f"{'='*80}\n")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_training_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_validation_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ============================================================================
# DATASET
# ============================================================================

class BUSIDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask

# ============================================================================
# DATA LOADING
# ============================================================================

def find_image_mask_pairs(root_dir: Path):
    print(f"{'='*80}")
    print(f"DATA LOADING")
    print(f"{'='*80}")
    print(f"Root: {root_dir}")
    
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    # Find images and masks directories
    images_dir, masks_dir = None, None
    for img_folder, mask_folder in [('images', 'masks'), ('Images', 'Masks')]:
        img_path = root_dir / img_folder
        mask_path = root_dir / mask_folder
        if img_path.exists() and mask_path.exists():
            images_dir, masks_dir = img_path, mask_path
            print(f"✓ Found: {img_folder}/ and {mask_folder}/")
            break
    
    if images_dir is None:
        images_dir = masks_dir = root_dir
        print(f"Using flat structure")
    
    # Collect image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.PNG', '*.JPG']:
        image_files.extend(list(images_dir.glob(ext)))
    
    image_files = sorted([f for f in image_files if 'mask' not in f.stem.lower()])
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    
    # Match masks
    image_paths, mask_paths = [], []
    for img_path in image_files:
        img_name = img_path.stem
        mask_found = None
        for pattern in [f"{img_name}_mask*.png", f"{img_name}_mask*.jpg"]:
            matches = list(masks_dir.glob(pattern))
            if matches:
                mask_found = matches[0]
                break
        
        if mask_found:
            image_paths.append(str(img_path))
            mask_paths.append(str(mask_found))
    
    print(f"Matched {len(image_paths)}/{len(image_files)} pairs")
    
    if len(image_paths) == 0:
        raise ValueError("No valid pairs found")
    
    return image_paths, mask_paths

def create_dataloaders(root: str, batch_size: int, val_ratio: float, test_ratio: float, seed: int):
    root_path = Path(root)
    image_paths, mask_paths = find_image_mask_pairs(root_path)
    
    # Split data
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=(val_ratio + test_ratio), random_state=seed
    )
    
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=test_ratio/(val_ratio + test_ratio), random_state=seed
    )
    
    print(f"Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
    
    # Create datasets
    train_dataset = BUSIDataset(train_imgs, train_masks, get_training_transforms())
    val_dataset = BUSIDataset(val_imgs, val_masks, get_validation_transforms())
    test_dataset = BUSIDataset(test_imgs, test_masks, get_validation_transforms())
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    
    print(f"✓ DataLoaders ready")
    print(f"{'='*80}\n")
    
    return train_loader, val_loader, test_loader

# ============================================================================
# MODEL
# ============================================================================

class AdaptiveUNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        
        print("Creating model...")
        
        # Load encoder
        efficientnet = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.encoder = efficientnet.features
        
        # FIXED: Define skip_indices BEFORE calling _detect_channels
        self.skip_indices = [2, 3, 5, 8]
        
        # Detect channels
        print("Detecting encoder dimensions...")
        skip_channels, bottleneck_channels = self._detect_channels()
        
        print(f"  Skip channels: {skip_channels}")
        print(f"  Bottleneck: {bottleneck_channels}")
        
        # Build decoder
        self.dec4 = nn.Sequential(
            nn.Conv2d(bottleneck_channels + skip_channels[3], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + skip_channels[2], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + skip_channels[1], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + skip_channels[0], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(32, num_classes, 1)
        
        print("✓ Model ready")
    
    def _detect_channels(self):
        with torch.no_grad():
            x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            skip_channels = []
            
            for idx, layer in enumerate(self.encoder):
                x = layer(x)
                if idx in self.skip_indices:
                    skip_channels.append(x.shape[1])
            
            bottleneck_channels = x.shape[1]
        
        return skip_channels, bottleneck_channels
    
    def forward(self, x):
        original_size = x.shape[2:]
        
        # Encoder
        skips = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in self.skip_indices:
                skips.append(x)
        
        # Decoder
        x = F.interpolate(x, size=skips[3].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.dec4(x)
        
        x = F.interpolate(x, size=skips[2].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.dec3(x)
        
        x = F.interpolate(x, size=skips[1].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.dec2(x)
        
        x = F.interpolate(x, size=skips[0].shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.dec1(x)
        
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        x = self.final(x)
        
        return x

# ============================================================================
# LOSS
# ============================================================================

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        pred_sigmoid = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal = alpha * ((1 - p_t) ** gamma) * ce
        return focal.mean()
    
    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def forward(self, pred, target):
        target = target.float()
        return 0.5 * self.focal_loss(pred, target) + 0.5 * self.dice_loss(pred, target)

# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate TP, FP, FN, TN
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    
    eps = 1e-7
    
    # Core metrics
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    
    # NEW METRICS
    specificity = (tn + eps) / (tn + fp + eps)  # True Negative Rate
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)  # Overall accuracy
    jaccard = iou  # Jaccard is same as IoU
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'jaccard': jaccard.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'accuracy': accuracy.item()
    }

# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Use torch.cuda.amp.autocast for older PyTorch versions
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = []
    
    pbar = tqdm(loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        metrics = calculate_metrics(outputs, masks)
        all_metrics.append(metrics)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{metrics["dice"]:.4f}'})
    
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    return total_loss / len(loader), avg_metrics

# ============================================================================
# MAIN
# ============================================================================


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def save_training_curves(train_losses, val_losses, metrics_history, save_dir):
    """Save comprehensive training curves as PNG"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice
    dice_scores = [m['dice'] for m in metrics_history]
    axes[0, 1].plot(epochs, dice_scores, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU/Jaccard
    iou_scores = [m['iou'] for m in metrics_history]
    axes[0, 2].plot(epochs, iou_scores, 'cyan', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('IoU/Jaccard')
    axes[0, 2].set_title('IoU/Jaccard Index')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision
    prec_scores = [m['precision'] for m in metrics_history]
    axes[0, 3].plot(epochs, prec_scores, 'purple', linewidth=2)
    axes[0, 3].set_xlabel('Epoch')
    axes[0, 3].set_ylabel('Precision')
    axes[0, 3].set_title('Precision')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Recall
    recall_scores = [m['recall'] for m in metrics_history]
    axes[1, 0].plot(epochs, recall_scores, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall (Sensitivity)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Specificity
    spec_scores = [m['specificity'] for m in metrics_history]
    axes[1, 1].plot(epochs, spec_scores, 'brown', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Specificity')
    axes[1, 1].set_title('Specificity')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy
    acc_scores = [m['accuracy'] for m in metrics_history]
    axes[1, 2].plot(epochs, acc_scores, 'pink', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 3].axis('off')
    best_epoch = dice_scores.index(max(dice_scores)) + 1
    summary_text = f"""
    Best Performance
    
    Epoch: {best_epoch}
    Dice: {max(dice_scores):.4f}
    Precision: {prec_scores[best_epoch-1]:.4f}
    Recall: {recall_scores[best_epoch-1]:.4f}
    Specificity: {spec_scores[best_epoch-1]:.4f}
    Accuracy: {acc_scores[best_epoch-1]:.4f}
    """
    axes[1, 3].text(0.1, 0.5, summary_text, fontsize=12, 
                   verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves: {save_path}")

def save_predictions(model, loader, device, save_dir, num_samples=8):
    """Save prediction visualizations as PNG"""
    model.eval()
    
    # Get samples
    images_list, masks_list, preds_list = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            images_list.append(images.cpu())
            masks_list.append(masks.cpu())
            preds_list.append(preds.cpu())
            
            if len(images_list) * images.size(0) >= num_samples:
                break
    
    images_all = torch.cat(images_list)[:num_samples]
    masks_all = torch.cat(masks_list)[:num_samples]
    preds_all = torch.cat(preds_list)[:num_samples]
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Denormalize image
        img = images_all[idx].numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = masks_all[idx, 0].numpy()
        pred = preds_all[idx, 0].numpy()
        
        # Original
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')
        
        # Ground truth
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # Prediction
        axes[idx, 2].imshow(pred, cmap='gray')
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')
        
        # Overlay
        axes[idx, 3].imshow(img)
        axes[idx, 3].imshow(pred, cmap='Reds', alpha=0.5)
        axes[idx, 3].set_title('Overlay')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions: {save_path}")

def save_metrics_table(metrics, save_dir, filename='final_metrics.png'):
    """Save metrics as a table PNG"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    data = [[k.capitalize(), f"{v:.4f}"] for k, v in metrics.items()]
    
    table = ax.table(cellText=data, colLabels=['Metric', 'Value'],
                    cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics table: {save_path}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def save_comprehensive_curves(train_losses, val_losses, metrics_history, save_dir):
    """Save comprehensive training curves with all 7 metrics"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Training Progress - All Metrics', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice Score
    dice_scores = [m['dice'] for m in metrics_history]
    axes[0, 1].plot(epochs, dice_scores, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Dice Score', fontsize=12)
    axes[0, 1].set_title('Dice Score', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=max(dice_scores), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(dice_scores):.4f}')
    axes[0, 1].legend(fontsize=10)
    
    # IoU/Jaccard
    iou_scores = [m['iou'] for m in metrics_history]
    axes[0, 2].plot(epochs, iou_scores, 'cyan', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('IoU/Jaccard', fontsize=12)
    axes[0, 2].set_title('IoU/Jaccard Index', fontsize=13, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision
    prec_scores = [m['precision'] for m in metrics_history]
    axes[0, 3].plot(epochs, prec_scores, 'purple', linewidth=2)
    axes[0, 3].set_xlabel('Epoch', fontsize=12)
    axes[0, 3].set_ylabel('Precision', fontsize=12)
    axes[0, 3].set_title('Precision', fontsize=13, fontweight='bold')
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='Target: 0.90')
    axes[0, 3].legend(fontsize=10)
    
    # Recall
    recall_scores = [m['recall'] for m in metrics_history]
    axes[1, 0].plot(epochs, recall_scores, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Recall', fontsize=12)
    axes[1, 0].set_title('Recall (Sensitivity)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.84, color='r', linestyle='--', alpha=0.5, label='Target: 0.84')
    axes[1, 0].legend(fontsize=10)
    
    # Specificity
    spec_scores = [m['specificity'] for m in metrics_history]
    axes[1, 1].plot(epochs, spec_scores, 'brown', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Specificity', fontsize=12)
    axes[1, 1].set_title('Specificity', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy
    acc_scores = [m['accuracy'] for m in metrics_history]
    axes[1, 2].plot(epochs, acc_scores, 'magenta', linewidth=2)
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Accuracy', fontsize=12)
    axes[1, 2].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Summary panel
    axes[1, 3].axis('off')
    best_epoch = dice_scores.index(max(dice_scores)) + 1
    summary_text = f"""Best Performance
    
Epoch: {best_epoch}
Dice: {max(dice_scores):.4f}
IoU/Jaccard: {iou_scores[best_epoch-1]:.4f}
Precision: {prec_scores[best_epoch-1]:.4f}
Recall: {recall_scores[best_epoch-1]:.4f}
Specificity: {spec_scores[best_epoch-1]:.4f}
Accuracy: {acc_scores[best_epoch-1]:.4f}
    """
    axes[1, 3].text(0.1, 0.5, summary_text, fontsize=11, 
                   verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves_all_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comprehensive training curves: {save_path}")

def save_prediction_visualizations(model, loader, device, save_dir, num_samples=12):
    """Save prediction visualizations"""
    model.eval()
    
    images_list, masks_list, preds_list = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            images_list.append(images.cpu())
            masks_list.append(masks.cpu())
            preds_list.append(preds.cpu())
            
            if len(images_list) * images.size(0) >= num_samples:
                break
    
    images_all = torch.cat(images_list)[:num_samples]
    masks_all = torch.cat(masks_list)[:num_samples]
    preds_all = torch.cat(preds_list)[:num_samples]
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Denormalize image
        img = images_all[idx].numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = masks_all[idx, 0].numpy()
        pred = preds_all[idx, 0].numpy()
        
        # Original
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'Sample {idx+1}: Original', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Ground truth
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Prediction
        axes[idx, 2].imshow(pred, cmap='gray')
        axes[idx, 2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Overlay
        axes[idx, 3].imshow(img)
        axes[idx, 3].imshow(pred, cmap='Reds', alpha=0.5)
        axes[idx, 3].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'predictions_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions visualization: {save_path}")

def save_metrics_table_image(metrics, save_dir, filename='metrics_table.png'):
    """Save metrics as a formatted table image"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    data = [
        ['Dice Score', f"{metrics['dice']:.4f}"],
        ['IoU', f"{metrics['iou']:.4f}"],
        ['Jaccard Index', f"{metrics['jaccard']:.4f}"],
        ['Precision', f"{metrics['precision']:.4f}"],
        ['Recall (Sensitivity)', f"{metrics['recall']:.4f}"],
        ['Specificity', f"{metrics['specificity']:.4f}"],
        ['Accuracy', f"{metrics['accuracy']:.4f}"]
    ]
    
    table = ax.table(cellText=data, colLabels=['Metric', 'Value'],
                    cellLoc='left', loc='center', colWidths=[0.7, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)
    
    # Alternate row colors
    for i in range(1, len(data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            table[(i, j)].set_text_props(fontsize=13)
    
    plt.title('Performance Metrics - Test Set', fontsize=16, fontweight='bold', pad=20)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics table: {save_path}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Model will be saved directly in SAVE_DIR
    
    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        DATA_ROOT, BATCH_SIZE, VAL_RATIO, TEST_RATIO, SEED
    )
    
    # Create model
    print(f"\n{'='*80}")
    print(f"MODEL SETUP")
    print(f"{'='*80}")
    
    model = AdaptiveUNet(num_classes=1).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = CombinedLoss()
    
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': BASE_LR * 0.1},
        {'params': decoder_params, 'lr': BASE_LR}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = GradScaler()
    
    print(f"✓ Setup complete\n")
    
    # Training
    print(f"{'='*80}")
    print(f"TRAINING")
    print(f"{'='*80}\n")
    
    best_dice = 0.0
    train_losses, val_losses, metrics_history = [], [], []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 80)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_history.append(val_metrics)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val Dice  : {val_metrics['dice']:.4f}")
        print(f"Val IoU   : {val_metrics['iou']:.4f}")
        print(f"Jaccard   : {val_metrics['jaccard']:.4f}")
        print(f"Precision : {val_metrics['precision']:.4f}")
        print(f"Recall    : {val_metrics['recall']:.4f}")
        print(f"Specificity: {val_metrics['specificity']:.4f}")
        print(f"Accuracy  : {val_metrics['accuracy']:.4f}")
        
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"✓ Saved best model (Dice: {best_dice:.4f})")
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'epoch_{epoch+1}.pth'))
    
    # Test
    print(f"\n{'='*80}")
    print(f"TESTING")
    print(f"{'='*80}")
    
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    test_loss, test_metrics = validate(model, test_loader, criterion, DEVICE)
    
    print(f"\nFINAL RESULTS:")
    print(f"Dice      : {test_metrics['dice']:.4f}")
    print(f"IoU       : {test_metrics['iou']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    
    with open(os.path.join(SAVE_DIR, 'results.txt'), 'w') as f:
        f.write(f"Dice: {test_metrics['dice']:.4f}\n")
        f.write(f"IoU: {test_metrics['iou']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"Best Val Dice: {best_dice:.4f}\n")
    
    # Generate comprehensive visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Save comprehensive training curves
    save_comprehensive_curves(train_losses, val_losses, metrics_history, SAVE_DIR)
    
    # Save prediction visualizations
    save_prediction_visualizations(model, test_loader, DEVICE, SAVE_DIR, num_samples=12)
    
    # Load best model for final test evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = validate(model, test_loader, criterion, DEVICE)
    
    print("\nTest Set Results:")
    print("="*80)
    print(f"{'Dice Score':<20s}: {test_metrics['dice']:.4f}")
    print(f"{'IoU':<20s}: {test_metrics['iou']:.4f}")
    print(f"{'Jaccard Index':<20s}: {test_metrics['jaccard']:.4f}")
    print(f"{'Precision':<20s}: {test_metrics['precision']:.4f}")
    print(f"{'Recall':<20s}: {test_metrics['recall']:.4f}")
    print(f"{'Specificity':<20s}: {test_metrics['specificity']:.4f}")
    print(f"{'Accuracy':<20s}: {test_metrics['accuracy']:.4f}")
    print("="*80)
    
    # Save test metrics table
    save_metrics_table_image(test_metrics, SAVE_DIR, 'test_metrics_table.png')
    
    print(f"\n{'='*80}")
    print(f"COMPLETE! Best Dice: {best_dice:.4f}")
    print(f"Results: {SAVE_DIR}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()