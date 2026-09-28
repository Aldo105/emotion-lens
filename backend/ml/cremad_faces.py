"""
Face crops from CREMA-D for training, from actors the evaluation never sees.

data/test_videos/evaluate_cremad.py scores the model on 20 CREMA-D actors
(listed in data/test_videos/cremad/indice.json). Training on any of them
would make that number meaningless, so they are excluded here and again
when the manifest is loaded (train_emotion_cnn.ManifestDataset).

Labels are CREMA-D's FaceVote -- what raters perceived from the face alone --
not the emotion the actor was asked to portray. Clips where raters split
("A:S") are skipped. Frames come from 40-80% of each clip: the onset is
still neutral and the expression peaks mid-sentence.

Licence: ODbL 1.0 / DbCL 1.0. Downloads stay in data/training/ (git-ignored).

Usage:
    python -m backend.ml.cremad_faces --max-actors 2     # smoke test
    python -m backend.ml.cremad_faces --delete-videos    # full run
"""
import argparse
import csv
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

import cv2

from backend.app.services.emotion_classifier import FER7_LABELS, EmotionClassifier

ROOT = Path(__file__).resolve().parents[2]
EVAL_INDEX = ROOT / "data" / "test_videos" / "cremad" / "indice.json"
OUT = ROOT / "data" / "training" / "cremad"
MANIFEST = OUT / "manifest.csv"
SUMMARY_URL = ("https://raw.githubusercontent.com/CheyneyComputerScience/"
               "CREMA-D/master/processedResults/summaryTable.csv")
VIDEO_BASE = ("https://media.githubusercontent.com/media/"
              "CheyneyComputerScience/CREMA-D/master/VideoFlash")
FACE_VOTE_TO_FER7 = {"A": "angry", "D": "disgust", "F": "fear",
                     "H": "happy", "N": "neutral", "S": "sad"}
FIELDS = ["ruta", "label", "actor", "split", "clip"]


def face_vote_label(vote: str) -> int | None:
    name = FACE_VOTE_TO_FER7.get(vote.strip())
    return FER7_LABELS.index(name) if name else None


def eval_actors(index_path) -> set[str]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} missing -- run data/test_videos/download_cremad.py --actores 20 "
            "first, otherwise evaluation actors cannot be excluded")
    clips = json.loads(index_path.read_text(encoding="utf-8"))["clips"]
    return {c["actor"] for c in clips}


def split_actors(actors, excluded, n_val=10, seed=0):
    pool = sorted(set(actors) - set(excluded))
    random.Random(seed).shuffle(pool)
    return sorted(pool[n_val:]), sorted(pool[:n_val])


def pick_frames(n_frames: int, k: int = 5, start: float = 0.4, end: float = 0.8) -> list[int]:
    if n_frames <= 0:
        return []
    a = int(n_frames * start)
    b = max(a + 1, int(n_frames * end))
    if b - a <= k:
        return list(range(a, b))
    step = (b - a) / k
    return [a + int(i * step) for i in range(k)]


def read_summary(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{"clip": r["FileName"], "actor": r["FileName"].split("_")[0],
                 "label": face_vote_label(r["FaceVote"])}
                for r in csv.DictReader(f)]


def pending_clips(clips: list[dict], manifest_rows: list[dict]) -> list[dict]:
    done = {r["clip"] for r in manifest_rows}
    return [c for c in clips if c["clip"] not in done]


def verify_manifest(rows, excluded: set[str]) -> None:
    leaked = sorted({r["actor"] for r in rows} & set(excluded))
    if leaked:
        raise ValueError(f"manifest contains evaluation actors: {leaked}")


def _read_frames(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _download(url: str, dest: Path) -> None:
    """Download a file atomically to avoid truncated files on interruption.

    Downloads to dest.with_name(dest.name + ".part"), then atomically
    replaces dest on success. If any exception occurs, the .part file is
    removed and the exception is re-raised.
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, part)
        os.replace(part, dest)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def build(max_actors=None, frames_per_clip=5, delete_videos=False):
    from backend.app.services.face_detector import FaceDetector

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "caras").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    summary_path = OUT / "summaryTable.csv"
    if not summary_path.exists():
        _download(SUMMARY_URL, summary_path)

    clips = [c for c in read_summary(summary_path) if c["label"] is not None]
    excluded = eval_actors(EVAL_INDEX)
    train, val = split_actors({c["actor"] for c in clips}, excluded)
    if max_actors:
        train, val = train[:max_actors], val[:1]
    split_of = {a: "train" for a in train} | {a: "val" for a in val}
    clips = [c for c in clips if c["actor"] in split_of]

    existing = []
    if MANIFEST.exists():
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    verify_manifest(existing, excluded)
    todo = pending_clips(clips, existing)
    print(f"[INFO] actors train={len(train)} val={len(val)} excluded={len(excluded)} "
          f"clips pending={len(todo)} done={len({r['clip'] for r in existing})}")

    det = FaceDetector(max_faces=1, detection_confidence=0.5, tracking_confidence=0.5)
    new_file = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, c in enumerate(todo, 1):
            video = OUT / "videos" / f"{c['clip']}.flv"
            if not video.exists():
                try:
                    _download(f"{VIDEO_BASE}/{c['clip']}.flv", video)
                except Exception as e:  # a missing clip must not kill a 6000-clip run
                    print(f"[WARN] {c['clip']}: {e}")
                    continue
            frames = _read_frames(video)
            for idx in pick_frames(len(frames), k=frames_per_clip):
                d = det.detect(frames[idx])
                gray = EmotionClassifier.crop_face_gray(frames[idx], d["bbox"]) if d else None
                if gray is None:
                    continue
                rel = f"caras/{c['clip']}_{idx:03d}.png"
                cv2.imwrite(str(OUT / rel), gray)
                writer.writerow({"ruta": rel, "label": c["label"], "actor": c["actor"],
                                 "split": split_of[c["actor"]], "clip": c["clip"]})
            f.flush()
            if delete_videos:
                video.unlink(missing_ok=True)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    det.close()
    print(f"[OK] manifest -> {MANIFEST}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-actors", type=int, default=None)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--delete-videos", action="store_true")
    args = ap.parse_args(argv)
    build(args.max_actors, args.frames, args.delete_videos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
