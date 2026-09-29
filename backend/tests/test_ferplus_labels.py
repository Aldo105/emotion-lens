import pytest

from backend.ml.ferplus_labels import (
    check_join, index_from_sample_id, join_ok, labels_for_split, load_votes, majority_label,
)

COLS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "nf"]


def _votes(**kw):
    return {c: kw.get(c, 0) for c in COLS}


def test_majority_clear_winner_maps_to_fer7_index():
    assert majority_label(_votes(sadness=7, neutral=3)) == 4  # sad


def test_single_stray_votes_are_dropped_before_majority():
    # 5 sadness vs 4 neutral + 1 anger: stray anger vote removed -> 5/9 > half
    assert majority_label(_votes(sadness=5, neutral=4, anger=1)) == 4


def test_no_majority_returns_none():
    assert majority_label(_votes(sadness=4, neutral=4, anger=2)) is None


def test_contempt_unknown_and_not_a_face_are_dropped():
    assert majority_label(_votes(contempt=8, neutral=2)) is None
    assert majority_label(_votes(unknown=6, neutral=4)) is None
    assert majority_label(_votes(nf=10)) is None
    assert majority_label(_votes()) is None


def test_index_from_sample_id():
    assert index_from_sample_id("fer2013_006289") == 6289
    with pytest.raises(ValueError):
        index_from_sample_id("affectnet_12")


def test_load_votes_reads_header_names_not_positions(tmp_path):
    p = tmp_path / "fer2013new.csv"
    p.write_text("Usage,Image name,neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF\n"
                 "Training,fer0000000.png,4,0,0,1,5,0,0,0,0,0\n"
                 "PublicTest,,0,0,0,0,0,0,0,0,0,10\n", encoding="utf-8")
    rows = load_votes(p)
    assert rows[0]["usage"] == "Training"
    assert rows[0]["votes"]["anger"] == 5
    assert rows[1]["votes"]["nf"] == 10


def test_load_votes_tolerates_bom(tmp_path):
    p = tmp_path / "fer2013new.csv"
    p.write_text("﻿Usage,Image name,neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF\n"
                 "Training,fer0000000.png,4,0,0,1,5,0,0,0,0,0\n", encoding="utf-8")
    rows = load_votes(p)
    assert rows[0]["usage"] == "Training"
    assert rows[0]["votes"]["anger"] == 5


def _rows():
    return [
        {"usage": "Training", "votes": _votes(happiness=10)},
        {"usage": "Training", "votes": _votes(sadness=9, neutral=1)},
        {"usage": "PublicTest", "votes": _votes(nf=10)},
    ]


def test_labels_for_split_follows_sample_id():
    split = {"sample_id": ["fer2013_000001", "fer2013_000000"]}
    assert labels_for_split(split, _rows()) == [4, 3]


def test_check_join_passes_when_aligned():
    split = {"sample_id": ["fer2013_000000", "fer2013_000001", "fer2013_000002"],
             "usage": ["Training", "Training", "PublicTest"],
             "emotion": [3, 4, 6]}
    stats = check_join({"train": split}, _rows())
    assert stats["usage_match"] == 1.0
    assert stats["label_agreement"] == 1.0
    assert stats["dropped"] == pytest.approx(1 / 3)


def test_check_join_detects_misalignment():
    split = {"sample_id": ["fer2013_000002", "fer2013_000000", "fer2013_000001"],
             "usage": ["Training", "Training", "PublicTest"],
             "emotion": [3, 4, 6]}
    assert not join_ok(check_join({"train": split}, _rows()))
