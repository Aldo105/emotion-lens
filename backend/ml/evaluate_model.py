"""
EmotionLens -- Standalone Emotion CNN Evaluation

Evaluates whatever checkpoint currently sits at
backend/ml/saved_models/emotion_cnn.pt against the FER2013 test split and
writes a report -- decoupled from the (long-running) training loop, so a
final accuracy number is always available even if a training run gets
interrupted before its own end-of-run report block executes.

Usage:
    python -m backend.ml.evaluate_model
"""

import json
import os

import numpy as np
import torch

from backend.ml.train_emotion_cnn import (
    FER2013Dataset, FER7_LABELS, OUTPUT_MODEL_PATH, BATCH_SIZE,
    build_model, run_epoch,
)
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_REPORT_PATH = os.path.join(SCRIPT_DIR, "saved_models", "training_report.json")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(OUTPUT_MODEL_PATH):
        print(f"[ERROR] No checkpoint found at {OUTPUT_MODEL_PATH}")
        return

    print("[INFO] Loading FER2013 test split...")
    raw = load_dataset("abhilash88/fer2013-enhanced")
    test_ds = FER2013Dataset(raw["test"], train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(device)
    model.load_state_dict(torch.load(OUTPUT_MODEL_PATH, map_location=device, weights_only=True))
    criterion = nn.CrossEntropyLoss()

    # Overall accuracy + confusion matrix (per-class breakdown)
    model.eval()
    confusion = np.zeros((len(FER7_LABELS), len(FER7_LABELS)), dtype=np.int64)
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1

    test_acc = correct / total
    test_loss = total_loss / total
    per_class_acc = {
        FER7_LABELS[i]: float(confusion[i, i] / max(confusion[i].sum(), 1))
        for i in range(len(FER7_LABELS))
    }

    print(f"[RESULT] test_acc={test_acc:.4f} test_loss={test_loss:.4f}")
    print("[RESULT] Per-class accuracy:")
    for label, acc in per_class_acc.items():
        print(f"    {label:10s} {acc:.3f}")

    report = {
        "labels": FER7_LABELS,
        "dataset": "abhilash88/fer2013-enhanced",
        "test_acc": test_acc,
        "test_loss": test_loss,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion.tolist(),
    }
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Report saved to {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
