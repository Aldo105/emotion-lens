"""
EmotionLens -- Emotion CNN Training Script

Fine-tunes an ImageNet-pretrained MobileNetV2 on FER2013 (7 basic emotions:
angry, disgust, fear, happy, sad, surprise, neutral -- see FER7_LABELS in
backend/app/services/emotion_classifier.py). This label order matches the
dataset's own ClassLabel order, so no remapping is needed.

"nervousness" and "confidence" (the other 2 of the app's 9 emotion_labels)
have no equivalent in any public FER dataset -- they stay derived from
Action Units / blendshapes, as already implemented in emotion_classifier.py.
This script only trains the 7-class model; backend/app/services/emotion_classifier.py
is responsible for merging its output with the heuristic nervousness/confidence
scores at inference time.

Dataset: abhilash88/fer2013-enhanced (HuggingFace Hub, Apache-2.0, no login
required) -- a cleaned FER2013 mirror with train/validation/test splits and
per-sample quality metadata.

Usage:
    python -m backend.ml.train_emotion_cnn
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────────────
FER7_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 15
PATIENCE = 4            # early stopping on val accuracy
LR = 1e-4
WEIGHT_DECAY = 1e-4
# 0 -- this machine only has ~8GB system RAM total; DataLoader worker
# processes previously pushed it into an OOM kill alongside the dev server.
NUM_WORKERS = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_models", "emotion_cnn.pt")
OUTPUT_REPORT_PATH = os.path.join(SCRIPT_DIR, "saved_models", "training_report.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FER2013Dataset(Dataset):
    """Wraps the HF FER2013 split; converts 48x48 grayscale -> 3ch tensor."""

    def __init__(self, hf_split, train: bool):
        self.ds = hf_split
        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = np.array(row["image"], dtype=np.uint8)  # (48, 48)
        label = int(row["emotion"])
        tensor = self.transform(img)
        return tensor, label


def build_model(device: str) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, len(FER7_LABELS))
    return model.to(device)


def compute_class_weights(hf_train_split) -> torch.Tensor:
    """Inverse-frequency class weights -- FER2013's 'disgust' class is
    heavily underrepresented (~1.5% of samples)."""
    counts = np.zeros(len(FER7_LABELS), dtype=np.int64)
    for label in hf_train_split["emotion"]:
        counts[label] += 1
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    print(f"[INFO] Class counts: {dict(zip(FER7_LABELS, counts.tolist()))}")
    print(f"[INFO] Class weights: {dict(zip(FER7_LABELS, np.round(weights, 3).tolist()))}")
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print("[INFO] Loading FER2013 (abhilash88/fer2013-enhanced) from HuggingFace Hub...")
    raw = load_dataset("abhilash88/fer2013-enhanced")

    train_ds = FER2013Dataset(raw["train"], train=True)
    val_ds = FER2013Dataset(raw["validation"], train=False)
    test_ds = FER2013Dataset(raw["test"], train=False)
    print(f"[INFO] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

    model = build_model(device)
    class_weights = compute_class_weights(raw["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []

    # Resume from a previous (possibly interrupted) run's checkpoint instead
    # of retraining from ImageNet weights -- re-evaluate it first so we know
    # its real val_acc and never overwrite it with a worse epoch.
    if os.path.exists(OUTPUT_MODEL_PATH):
        print(f"[INFO] Found existing checkpoint at {OUTPUT_MODEL_PATH} -- resuming from it")
        model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location=device, weights_only=True))
        _, resumed_val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        best_val_acc = resumed_val_acc
        print(f"[INFO] Resumed checkpoint val_acc={resumed_val_acc:.4f}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()
        epoch_time = time.time() - t0

        print(f"[EPOCH {epoch}/{MAX_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({epoch_time:.1f}s)")
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "epoch_seconds": epoch_time,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            print(f"[OK] New best val_acc={val_acc:.4f} -- saved to {OUTPUT_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[INFO] Early stopping -- no improvement in {PATIENCE} epochs")
                break

    # ── Final test evaluation using the BEST checkpoint ──────────────────
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location=device, weights_only=True))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, scaler, device, train=False)
    print(f"[FINAL] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    report = {
        "labels": FER7_LABELS,
        "dataset": "abhilash88/fer2013-enhanced",
        "img_size": IMG_SIZE,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "epochs_trained": len(history),
        "history": history,
    }
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Training report saved to {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
