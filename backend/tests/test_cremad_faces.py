import json
from unittest.mock import patch

import pytest

from backend.ml.cremad_faces import (
    _download, eval_actors, face_vote_label, pending_clips, pick_frames, read_summary,
    split_actors, verify_manifest,
)


def test_face_vote_label_maps_single_votes_and_rejects_ties():
    assert face_vote_label("S") == 4
    assert face_vote_label("N") == 6
    assert face_vote_label("A:S") is None
    assert face_vote_label("") is None


def test_eval_actors_reads_index(tmp_path):
    p = tmp_path / "indice.json"
    p.write_text(json.dumps({"clips": [{"actor": "1073"}, {"actor": "1002"}, {"actor": "1073"}]}))
    assert eval_actors(p) == {"1073", "1002"}


def test_eval_actors_missing_index_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_cremad"):
        eval_actors(tmp_path / "nope.json")


def test_eval_actors_empty_index_is_an_error(tmp_path):
    p = tmp_path / "indice.json"
    p.write_text(json.dumps({"clips": []}))
    with pytest.raises(ValueError, match="no clips"):
        eval_actors(p)


def test_split_actors_disjoint_deterministic_and_excludes():
    actors = [str(1001 + i) for i in range(91)]
    excluded = {"1002", "1019", "1020"}
    train, val = split_actors(actors, excluded, n_val=10, seed=0)
    assert len(val) == 10 and len(train) == 91 - 3 - 10
    assert not (set(train) & set(val))
    assert not ((set(train) | set(val)) & excluded)
    assert (train, val) == split_actors(actors, excluded, n_val=10, seed=0)


def test_pick_frames_middle_of_clip():
    assert pick_frames(100, k=5) == [40, 48, 56, 64, 72]
    assert pick_frames(5, k=5) == [2, 3]
    assert pick_frames(0) == []


def test_read_summary_parses_actor_and_face_vote(tmp_path):
    p = tmp_path / "summaryTable.csv"
    p.write_text('"","FileName","VoiceVote","VoiceLevel","FaceVote","FaceLevel","MultiModalVote","MultiModalLevel"\n'
                 '"1","1001_IEO_NEU_XX","N","69.1","N","92.22","N","64.78"\n'
                 '"2","1001_IEO_HAP_LO","N","71.67","H:N","57","H","57.38"\n', encoding="utf-8")
    assert read_summary(p) == [
        {"clip": "1001_IEO_NEU_XX", "actor": "1001", "label": 6},
        {"clip": "1001_IEO_HAP_LO", "actor": "1001", "label": None},
    ]


def test_pending_clips_skips_done():
    clips = [{"clip": "a"}, {"clip": "b"}, {"clip": "c"}]
    done = [{"clip": "b"}, {"clip": "b"}]
    assert pending_clips(clips, done) == [{"clip": "a"}, {"clip": "c"}]


def test_verify_manifest_rejects_eval_actor():
    verify_manifest([{"actor": "1001"}], {"1073"})
    with pytest.raises(ValueError, match="1073"):
        verify_manifest([{"actor": "1001"}, {"actor": "1073"}], {"1073"})


def test_download_cleans_up_part_file_on_failure(tmp_path):
    """_download should remove .part file and re-raise on urlretrieve failure."""
    dest = tmp_path / "data.bin"

    def fake_retrieve(url, target):
        raise RuntimeError("Network error")

    with patch("backend.ml.cremad_faces.urllib.request.urlretrieve", fake_retrieve):
        with pytest.raises(RuntimeError, match="Network error"):
            _download("http://example.com/file", dest)

    assert not dest.exists(), "dest should not exist after failed download"
    assert not (dest.with_name(dest.name + ".part")).exists(), ".part file should be cleaned up"


def test_download_succeeds_and_cleans_up_part_file(tmp_path):
    """_download should atomically move .part to dest on success."""
    dest = tmp_path / "data.bin"
    content = b"test data"

    def fake_retrieve(url, target):
        with open(target, "wb") as f:
            f.write(content)

    with patch("backend.ml.cremad_faces.urllib.request.urlretrieve", fake_retrieve):
        _download("http://example.com/file", dest)

    assert dest.exists(), "dest should exist after successful download"
    assert dest.read_bytes() == content, "dest should have the correct content"
    assert not (dest.with_name(dest.name + ".part")).exists(), ".part file should not remain"
