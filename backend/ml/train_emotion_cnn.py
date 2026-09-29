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
    python -m backend.ml.train_emotion_cnn --name <candidato> [--labels ferplus]
        [--extra-manifest data/training/cremad/manifest.csv] [--init-from X.pt]
        [--sad-weight-mult 1.5] [--num-workers 0] [--batch-size 32]
Writes backend/ml/saved_models/candidatos/<candidato>.pt -- never emotion_cnn.pt.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torchvision import models, transforms

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
CANDIDATES_DIR = os.path.join(SCRIPT_DIR, "saved_models", "candidatos")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def candidate_paths(name: str) -> tuple[str, str]:
    """Every run writes here -- promoting a candidate to OUTPUT_MODEL_PATH
    is a deliberate manual step after evaluate_cremad.py (see the plan)."""
    safe = os.path.basename(name.replace("\\", "/")) or "run"
    return (os.path.join(CANDIDATES_DIR, f"{safe}.pt"),
            os.path.join(CANDIDATES_DIR, f"{safe}.json"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train an emotion CNN candidate")
    p.add_argument("--name", required=True, help="candidate name -> saved_models/candidatos/<name>.pt")
    p.add_argument("--labels", choices=["fer", "ferplus"], default="fer")
    p.add_argument("--extra-manifest", default=None, help="manifest.csv from backend.ml.cremad_faces")
    p.add_argument("--init-from", default=None, help="checkpoint to start from instead of ImageNet")
    p.add_argument("--sad-weight-mult", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return p.parse_args(argv)


def make_transform(train: bool):
    steps = [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3),
             transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class FER2013Dataset(Dataset):
    """Wraps the HF FER2013 split; converts 48x48 grayscale -> 3ch tensor.
    `labels` overrides the split's own (e.g. FER+); None entries are dropped."""

    def __init__(self, hf_split, train: bool, labels=None):
        self.ds = hf_split
        if labels is None:
            labels = list(hf_split["emotion"])
        self.indices = [i for i, lbl in enumerate(labels) if lbl is not None]
        self.labels = [int(labels[i]) for i in self.indices]
        self.transform = make_transform(train)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.ds[self.indices[idx]]
        img = np.array(row["image"], dtype=np.uint8)  # (48, 48)
        return self.transform(img), self.labels[idx]


def load_eval_actors() -> set[str]:
    from backend.ml.cremad_faces import EVAL_INDEX, eval_actors
    return eval_actors(EVAL_INDEX)


def read_manifest(path, split: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class ManifestDataset(Dataset):
    """Face crops written by backend/ml/cremad_faces.py: 224x224 grayscale
    PNGs, cut with EmotionClassifier.crop_face_gray exactly like inference."""

    def __init__(self, manifest_path, split: str, train: bool):
        from backend.ml.cremad_faces import verify_manifest
        rows = read_manifest(manifest_path, split)
        if not rows:
            raise ValueError(f"manifest {manifest_path} has no rows for split {split!r}")
        verify_manifest(rows, load_eval_actors())
        base = os.path.dirname(os.path.abspath(manifest_path))
        self.paths = [os.path.join(base, r["ruta"]) for r in rows]
        self.labels = [int(r["label"]) for r in rows]
        self.transform = make_transform(train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # cv2 + torch + datasets in that order crashes on this env (DLL clash).
        img = np.array(Image.open(self.paths[idx]).convert("L"))
        return self.transform(img), self.labels[idx]


def build_model(device: str) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, len(FER7_LABELS))
    return model.to(device)


def class_weights_from_labels(labels, sad_mult: float = 1.0) -> torch.Tensor:
    """Inverse-frequency class weights -- 'disgust' is ~1.5% of FER2013.
    sad_mult deliberately over-weights 'sad', the class we agreed must work."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(FER7_LABELS))
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    weights[FER7_LABELS.index("sad")] *= sad_mult
    print(f"[INFO] Class counts: {dict(zip(FER7_LABELS, counts.tolist()))}")
    print(f"[INFO] Class weights: {dict(zip(FER7_LABELS, np.round(weights, 3).tolist()))}")
    return torch.tensor(weights, dtype=torch.float32)


def select_val_acc(fer_val_acc: float, extra_val_acc: float | None) -> float:
    """Early stopping follows the held-out CREMA-D actors when present: that
    is the domain the app runs in, FER2013 val is only a proxy."""
    return fer_val_acc if extra_val_acc is None else extra_val_acc


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


def main(argv=None):
    from datasets import load_dataset  # deferred: imports huggingface_hub -> ssl

    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    model_path, report_path = candidate_paths(args.name)
    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    print("[INFO] Loading FER2013 (abhilash88/fer2013-enhanced) from HuggingFace Hub...")
    raw = load_dataset("abhilash88/fer2013-enhanced")
    labels = {s: None for s in ("train", "validation", "test")}
    if args.labels == "ferplus":
        from backend.ml.ferplus_labels import (
            check_join, ensure_csv, join_ok, labels_for_split, load_votes)
        rows = load_votes(ensure_csv())
        stats = check_join(dict(raw), rows)
        print(f"[INFO] FER+ join: {stats}")
        if not join_ok(stats):
            raise SystemExit(f"[ERROR] FER+ join does not line up: {stats}")
        labels = {s: labels_for_split(raw[s], rows) for s in labels}

    train_parts = [FER2013Dataset(raw["train"], True, labels["train"])]
    fer_val = FER2013Dataset(raw["validation"], False, labels["validation"])
    fer_test = FER2013Dataset(raw["test"], False, labels["test"])
    extra_val = None
    if args.extra_manifest:
        train_parts.append(ManifestDataset(args.extra_manifest, "train", True))
        extra_val = ManifestDataset(args.extra_manifest, "val", False)
    train_ds = ConcatDataset(train_parts)
    print(f"[INFO] train={len(train_ds)} fer_val={len(fer_val)} "
          f"extra_val={len(extra_val) if extra_val else 0} fer_test={len(fer_test)}")

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=(device == "cuda"))

    train_loader = loader(train_ds, True)
    fer_val_loader, fer_test_loader = loader(fer_val, False), loader(fer_test, False)
    extra_val_loader = loader(extra_val, False) if extra_val else None

    model = build_model(device)
    all_labels = [lbl for part in train_parts for lbl in part.labels]
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(all_labels, args.sad_weight_mult).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def evaluate():
        _, fer_acc = run_epoch(model, fer_val_loader, criterion, optimizer, scaler, device, train=False)
        extra_acc = None
        if extra_val_loader:
            _, extra_acc = run_epoch(model, extra_val_loader, criterion, optimizer, scaler, device, train=False)
        return fer_acc, extra_acc

    best_val_acc, epochs_without_improvement, history = 0.0, 0, []
    # Resume an interrupted run of this same candidate; otherwise start from
    # --init-from or ImageNet. Re-evaluate so a worse epoch never overwrites it.
    start = model_path if os.path.exists(model_path) else args.init_from
    if args.init_from and start == model_path:
        print(f"[WARN] --init-from ignored: resuming existing candidate {model_path}")
    if start:
        print(f"[INFO] Starting from {start}")
        model.load_state_dict(torch.load(start, map_location=device, weights_only=True))
        if start == model_path:
            best_val_acc = select_val_acc(*evaluate())
            print(f"[INFO] Resumed val_acc={best_val_acc:.4f}")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        fer_acc, extra_acc = evaluate()
        val_acc = select_val_acc(fer_acc, extra_acc)
        scheduler.step()
        epoch_time = time.time() - t0
        print(f"[EPOCH {epoch}/{args.max_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"fer_val={fer_acc:.4f} extra_val={extra_acc} ({epoch_time:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "fer_val_acc": fer_acc, "extra_val_acc": extra_acc, "epoch_seconds": epoch_time})
        if val_acc > best_val_acc:
            best_val_acc, epochs_without_improvement = val_acc, 0
            torch.save(model.state_dict(), model_path)
            print(f"[OK] New best val_acc={val_acc:.4f} -- saved to {model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[INFO] Early stopping -- no improvement in {PATIENCE} epochs")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_loss, test_acc = run_epoch(model, fer_test_loader, criterion, optimizer, scaler, device, train=False)
    print(f"[FINAL] fer_test_acc={test_acc:.4f} (labels={args.labels}; not comparable across label sets)")
    report = {"args": vars(args), "labels": FER7_LABELS, "img_size": IMG_SIZE,
              "best_val_acc": best_val_acc, "fer_test_acc": test_acc, "fer_test_loss": test_loss,
              "epochs_trained": len(history), "history": history}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Report saved to {report_path}")


if __name__ == "__main__":
    main()
