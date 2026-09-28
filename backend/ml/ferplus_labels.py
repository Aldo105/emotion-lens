"""
FER+ labels for the FER2013 images already used in training.

FER2013 was labelled by one person per image; FER+ (Microsoft) re-labelled
the same images with 10 crowd taggers each. fer2013new.csv has one row per
row of the original fer2013.csv, in the same order. The HF mirror we train
from (abhilash88/fer2013-enhanced) re-splits the data, but its sample_id
("fer2013_006289") appears to encode that original row index. "Appears" is
not good enough to train on, so --verificar checks it before anything uses
these labels.

Usage:
    python -m backend.ml.ferplus_labels --verificar
"""
import argparse
import csv
import sys
import urllib.request
from pathlib import Path

from backend.app.services.emotion_classifier import FER7_LABELS

FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "training" / "ferplus" / "fer2013new.csv"

VOTE_COLUMNS = ["neutral", "happiness", "surprise", "sadness", "anger",
                "disgust", "fear", "contempt", "unknown", "nf"]
FERPLUS_TO_FER7 = {"anger": "angry", "disgust": "disgust", "fear": "fear",
                   "happiness": "happy", "sadness": "sad",
                   "surprise": "surprise", "neutral": "neutral"}


def majority_label(votes: dict[str, int]) -> int | None:
    """Microsoft's 'majority' mode: discard single stray votes, then keep the
    image only if one class holds more than half of what remains. contempt,
    unknown and not-a-face have no FER7 equivalent and are dropped."""
    cleaned = {k: (v if v > 1 else 0) for k, v in votes.items()}
    total = sum(cleaned.values())
    if total == 0:
        return None
    top = max(cleaned, key=cleaned.get)
    if cleaned[top] * 2 <= total:
        return None
    name = FERPLUS_TO_FER7.get(top)
    return FER7_LABELS.index(name) if name else None


def load_votes(csv_path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            norm = {k.strip().lower(): (v or "").strip() for k, v in raw.items()}
            rows.append({"usage": norm.get("usage", ""),
                         "votes": {c: int(norm.get(c) or 0) for c in VOTE_COLUMNS}})
    return rows


def index_from_sample_id(sample_id: str) -> int:
    prefix, _, num = sample_id.rpartition("_")
    if prefix != "fer2013" or not num.isdigit():
        raise ValueError(f"unexpected sample_id: {sample_id!r}")
    return int(num)


def labels_for_split(hf_split, rows) -> list[int | None]:
    return [majority_label(rows[index_from_sample_id(s)]["votes"])
            for s in hf_split["sample_id"]]


def check_join(hf_splits: dict, rows) -> dict:
    n = usage_ok = kept = agree = 0
    for split in hf_splits.values():
        for sid, usage, emotion in zip(split["sample_id"], split["usage"], split["emotion"]):
            row = rows[index_from_sample_id(sid)]
            n += 1
            usage_ok += int(row["usage"] == usage)
            label = majority_label(row["votes"])
            if label is not None:
                kept += 1
                agree += int(label == emotion)
    return {"n": n,
            "usage_match": usage_ok / n if n else 0.0,
            "label_agreement": agree / kept if kept else 0.0,
            "dropped": (n - kept) / n if n else 0.0}


def join_ok(stats: dict) -> bool:
    # A correct join puts every image in its original Usage split, and FER+
    # re-labels only a minority of images; a shuffled join agrees at chance.
    return stats["usage_match"] >= 0.999 and 0.5 <= stats["label_agreement"] <= 0.97


def ensure_csv() -> Path:
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading {FERPLUS_URL}")
        urllib.request.urlretrieve(FERPLUS_URL, CSV_PATH)
    return CSV_PATH


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", action="store_true")
    args = ap.parse_args(argv)
    if not args.verificar:
        ap.print_help()
        return 0
    from datasets import load_dataset
    raw = load_dataset("abhilash88/fer2013-enhanced")
    rows = load_votes(ensure_csv())
    print(f"[INFO] fer2013new.csv rows={len(rows)}")
    stats = check_join(dict(raw), rows)
    print(f"[INFO] {stats}")
    if not join_ok(stats):
        print("[ERROR] FER+ join does not line up -- do NOT train with --labels ferplus")
        return 1
    print("[OK] FER+ join verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
