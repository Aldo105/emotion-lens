# backend/tests/test_train_helpers.py
import cv2
import numpy as np
import pytest

from backend.ml import train_emotion_cnn as T


class FakeSplit:
    def __init__(self, emotions):
        self.rows = [{"image": np.full((48, 48), 128, np.uint8), "emotion": e} for e in emotions]

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        return self.rows[key]

    def __len__(self):
        return len(self.rows)


def test_candidate_path_never_production():
    for name in ("emotion_cnn", "../emotion_cnn", "x"):
        pt, js = T.candidate_paths(name)
        assert pt != T.OUTPUT_MODEL_PATH
        assert "candidatos" in pt and pt.endswith(".pt") and js.endswith(".json")


def test_class_weights_inverse_frequency_and_sad_multiplier():
    labels = [0] * 10 + [4] * 5 + [3] * 5 + [1, 2, 5, 6]
    w = T.class_weights_from_labels(labels)
    assert w[0] < w[4]
    w_sad = T.class_weights_from_labels(labels, sad_mult=2.0)
    assert float(w_sad[4]) == pytest.approx(2 * float(w[4]))
    assert float(w_sad[0]) == pytest.approx(float(w[0]))


def test_fer_dataset_drops_none_labels():
    ds = T.FER2013Dataset(FakeSplit([0, 1, 2]), train=False, labels=[5, None, 3])
    assert len(ds) == 2
    assert ds.labels == [5, 3]
    x, y = ds[1]
    assert tuple(x.shape) == (3, 224, 224) and y == 3


def _manifest(tmp_path):
    (tmp_path / "caras").mkdir()
    cv2.imwrite(str(tmp_path / "caras" / "a.png"), np.full((224, 224), 90, np.uint8))
    cv2.imwrite(str(tmp_path / "caras" / "b.png"), np.full((224, 224), 90, np.uint8))
    p = tmp_path / "manifest.csv"
    p.write_text("ruta,label,actor,split,clip\n"
                 "caras/a.png,4,1001,train,1001_DFA_SAD_XX\n"
                 "caras/b.png,3,1002,val,1002_DFA_HAP_XX\n", encoding="utf-8")
    return p


def test_read_manifest_filters_split(tmp_path):
    rows = T.read_manifest(_manifest(tmp_path), "train")
    assert [r["actor"] for r in rows] == ["1001"]


def test_manifest_dataset_loads_gray_png_as_3ch(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "load_eval_actors", lambda: set())
    ds = T.ManifestDataset(_manifest(tmp_path), "train", train=False)
    x, y = ds[0]
    assert tuple(x.shape) == (3, 224, 224) and y == 4


def test_manifest_dataset_refuses_eval_actor(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "load_eval_actors", lambda: {"1001"})
    with pytest.raises(ValueError, match="1001"):
        T.ManifestDataset(_manifest(tmp_path), "train", train=False)


def test_manifest_dataset_empty_split_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "load_eval_actors", lambda: set())
    with pytest.raises(ValueError, match="no rows"):
        T.ManifestDataset(_manifest(tmp_path), "test", train=False)


def test_select_val_acc_prefers_target_domain():
    assert T.select_val_acc(0.7, None) == 0.7
    assert T.select_val_acc(0.7, 0.4) == 0.4


def test_parse_args_defaults():
    a = T.parse_args(["--name", "x"])
    assert (a.labels, a.extra_manifest, a.sad_weight_mult, a.num_workers) == ("fer", None, 1.0, 0)
