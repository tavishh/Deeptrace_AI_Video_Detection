"""
train.py - DeepTrace training loop

Owner: Zihao/Tavish
Week:  3

Trains the MLP head on top of frozen CLIP features using combined
Celeb-DF v2 (face-swap) and DeepTrace-GV (whole-scene AI) clips.
Joint training on both manipulation types is the key to cross-generator
generalization.

Usage:
    python src/model/train.py --config configs/config.yaml
    python src/model/train.py --config configs/config.yaml --logo-held-out sora
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Optional
 
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional WandB - gracefully disabled if not configured
# ---------------------------------------------------------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
 

# ---------------------------------------------------------------------------
# CLIP preprocessing (matches ViT-B/32 training)
# ---------------------------------------------------------------------------
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])
 
AUGMENT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
 
class VideoClipDataset(Dataset):
    """
    Loads video clips and extracts frames on the fly.
    Each __getitem__ returns one randomly sampled frame per clip.
    """
 
    def __init__(self, samples: list, transform, frames_per_clip: int = 5,
                 augment: bool = False):
        """
        Args:
            samples:         List of (clip_path, label) tuples. label: 0=real, 1=fake
            transform:       Torchvision transform to apply to each frame
            frames_per_clip: Number of frames to sample per clip per epoch
            augment:         Whether to apply augmentation
        """
        self.samples         = samples
        self.transform       = transform
        self.frames_per_clip = frames_per_clip
        self.augment         = augment
 
    def _sample_frame(self, clip_path: str) -> Optional[Image.Image]:
        """Sample one random frame from a video clip."""
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        idx = random.randint(0, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        img = self._sample_frame(clip_path)
        if img is None:
            tensor = torch.zeros(3, 224, 224)
        else:
            transform = AUGMENT_TRANSFORM if self.augment else self.transform
            tensor = transform(img)
        return tensor, label
 

 # ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
 
def load_deeptrace_samples(manifest_path: str, deeptrace_root: str,
                           split: str, mode: str,
                           logo_held_out: Optional[str] = None) -> list:
    """
    Load DeepTrace-GV samples from manifest.
 
    Args:
        manifest_path:   Path to staging_manifest.csv
        deeptrace_root:  Root directory of reencoded clips
        split:           train | val | test
        mode:            faceswap | ai_video | joint
        logo_held_out:   Generator to exclude from training (LOGO protocol)
 
    Returns:
        List of (clip_path, label) tuples
    """
    if mode == "faceswap":
        return []  # DeepTrace-GV not used in faceswap-only mode
 
    samples = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            generator = row["generator"]
 
            # LOGO: skip held-out generator from training
            if split == "train" and logo_held_out and generator == logo_held_out:
                continue
 
            clip_path = os.path.join(deeptrace_root, generator, f"{row['clip_id']}.mp4")
            if not os.path.exists(clip_path):
                continue
 
            label = 0 if row["label"] == "real" else 1
            samples.append((clip_path, label))
 
    return samples
 
 
def load_celebdf_samples(celebdf_root: str, train_ratio: float = 0.8) -> tuple:
    """
    Load Celeb-DF v2 samples from the test list.
    Since we only have the test split, we divide it into train/val.
 
    Returns:
        (train_samples, val_samples) - lists of (clip_path, label) tuples
    """
    list_path = os.path.join(celebdf_root, "List_of_testing_videos.txt")
    if not os.path.exists(list_path):
        print(f"[warning] Celeb-DF v2 not found at {celebdf_root}")
        return [], []
 
    samples = []
    with open(list_path) as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue
            label_raw = int(parts[0])
            rel_path  = parts[1]
            clip_path = os.path.join(celebdf_root, rel_path)
            if not os.path.exists(clip_path):
                continue
            label = 0 if label_raw == 1 else 1  # 0=real, 1=fake
            samples.append((clip_path, label))
 
    random.shuffle(samples)
    n_train = int(len(samples) * train_ratio)
    return samples[:n_train], samples[n_train:]
 
 
def build_dataloaders(manifest_path: str, deeptrace_root: str,
                      celebdf_root: str, mode: str,
                      batch_size: int = 32, num_workers: int = 2,
                      logo_held_out: Optional[str] = None) -> tuple:
    """
    Build train/val/test dataloaders for the specified training mode.
 
    Args:
        mode: faceswap | ai_video | joint
 
    Returns:
        train_loader, val_loader, test_loader
    """
    celebdf_train, celebdf_val = load_celebdf_samples(celebdf_root)
 
    dt_train = load_deeptrace_samples(manifest_path, deeptrace_root,
                                      "train", mode, logo_held_out)
    dt_val   = load_deeptrace_samples(manifest_path, deeptrace_root,
                                      "val",   mode, logo_held_out)
    dt_test  = load_deeptrace_samples(manifest_path, deeptrace_root,
                                      "test",  mode)
 
    if mode == "faceswap":
        train_samples = celebdf_train
        val_samples   = celebdf_val
        test_samples  = dt_test  # evaluate on DeepTrace-GV test
    elif mode == "ai_video":
        train_samples = dt_train
        val_samples   = dt_val
        test_samples  = dt_test
    else:  # joint
        train_samples = celebdf_train + dt_train
        val_samples   = celebdf_val   + dt_val
        test_samples  = dt_test
 
    random.shuffle(train_samples)
    random.shuffle(val_samples)
 
    print(f"\nDataset ({mode} mode):")
    print(f"  Train: {len(train_samples)} clips "
          f"(real={sum(1 for _,l in train_samples if l==0)}, "
          f"fake={sum(1 for _,l in train_samples if l==1)})")
    print(f"  Val:   {len(val_samples)} clips")
    print(f"  Test:  {len(test_samples)} clips")
 
    train_dataset = VideoClipDataset(train_samples, CLIP_TRANSFORM, augment=True)
    val_dataset   = VideoClipDataset(val_samples,   CLIP_TRANSFORM, augment=False)
    test_dataset  = VideoClipDataset(test_samples,  CLIP_TRANSFORM, augment=False)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
 
    return train_loader, val_loader, test_loader
 
 
# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
 
def train_epoch(model, loader, optimizer, criterion, device) -> dict:
    """Run one training epoch. Returns metrics dict."""
    model.head.train()
    model.extractor.eval()
 
    total_loss = 0
    all_probs, all_labels = [], []
 
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model.forward_frames(frames)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
 
    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
 
    return {"loss": avg_loss, "auc": auc}
 
 
def evaluate(model, loader, device) -> dict:
    """Run evaluation. Returns metrics dict."""
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
 
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            logits = model.forward_frames(frames)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
 
    avg_loss = total_loss / len(loader)
    preds    = (np.array(all_probs) > 0.5).astype(int)
 
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    acc = accuracy_score(all_labels, preds)
 
    return {"loss": avg_loss, "auc": auc, "acc": acc}
 
 
# ---------------------------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------------------------
 
def train(args) -> None:
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
 
    # WandB
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="deeptrace",
            name=f"{args.mode}_{args.logo_held_out or 'full'}",
            config=vars(args),
            tags=["clip", "mlp", args.mode],
        )
 
    # Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        manifest_path  = args.manifest,
        deeptrace_root = args.deeptrace_root,
        celebdf_root   = args.celebdf_root,
        mode           = args.mode,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        logo_held_out  = args.logo_held_out,
    )
 
    if len(train_loader) == 0:
        print("ERROR: No training data found. Check paths.")
        return
 
    # Model
    from clip_extractor import CLIPExtractor
    from mlp_head import MLPHead, DeepTrace
 
    extractor = CLIPExtractor().to(device)
    extractor.eval()
    head      = MLPHead(
        hidden_dim  = args.mlp_hidden_dim,
        dropout     = args.mlp_dropout,
    ).to(device)
    model = DeepTrace(extractor, head, aggregation=args.aggregation).to(device)
 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
 
    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()
 
    # Checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name       = f"{args.mode}_{args.logo_held_out or 'full'}"
    best_ckpt_path = checkpoint_dir / f"deeptrace_{run_name}_best.pth"
 
    # Training loop
    best_val_auc   = 0.0
    patience_count = 0
 
    print(f"\nTraining ({args.mode} mode, {args.epochs} epochs)...\n")
 
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
 
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    criterion, device)
        val_metrics   = evaluate(model, val_loader, device)
        scheduler.step()
 
        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc']:.4f} | "
              f"Time: {elapsed:.1f}s")
 
        if use_wandb:
            wandb.log({
                "epoch":       epoch,
                "train/loss":  train_metrics["loss"],
                "train/auc":   train_metrics["auc"],
                "val/loss":    val_metrics["loss"],
                "val/auc":     val_metrics["auc"],
            })
 
        # Early stopping + checkpoint
        if val_metrics["auc"] > best_val_auc:
            best_val_auc   = val_metrics["auc"]
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "mode":        args.mode,
                "logo_held_out": args.logo_held_out,
                "val_auc":     best_val_auc,
                "head_state":  model.head.state_dict(),
                "args":        vars(args),
            }, best_ckpt_path)
            print(f"  Saved best model (val AUC={best_val_auc:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break
 
    # Final evaluation on test set
    print(f"\nLoading best model from {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.head.load_state_dict(ckpt["head_state"])
 
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest Results ({run_name}):")
    print(f"  AUC:      {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['acc']:.4f}")
 
    if use_wandb:
        wandb.log({
            "test/auc": test_metrics["auc"],
            "test/acc": test_metrics["acc"],
        })
        wandb.finish()
 
    print(f"\nDone. Best checkpoint: {best_ckpt_path}")
 
 
# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeepTrace MLP head on frozen CLIP features."
    )
 
    # Training mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["faceswap", "ai_video", "joint"],
                        help="Training mode: faceswap | ai_video | joint")
    parser.add_argument("--logo-held-out", type=str, default=None,
                        choices=["sora", "kling", "veo"],
                        help="LOGO protocol: hold out this generator from training")
 
    # Paths (override defaults for Colab)
    parser.add_argument("--manifest",       type=str,
                        default="data/manifests/staging_manifest.csv")
    parser.add_argument("--deeptrace-root", type=str,
                        default="data/reencoded")
    parser.add_argument("--celebdf-root",   type=str,
                        default="data/celebdf")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="models/checkpoints")
 
    # Hyperparameters
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--patience",       type=int,   default=5)
    parser.add_argument("--mlp-hidden-dim", type=int,   default=128)
    parser.add_argument("--mlp-dropout",    type=float, default=0.3)
    parser.add_argument("--aggregation",    type=str,   default="mean",
                        choices=["mean", "majority_vote"])
    parser.add_argument("--num-workers",    type=int,   default=2)
    parser.add_argument("--seed",           type=int,   default=42)
 
    # WandB
    parser.add_argument("--no-wandb",       action="store_true",
                        help="Disable WandB logging")
 
    args = parser.parse_args()
    train(args)