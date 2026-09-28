import json

import pytest

from backend.ml.cremad_faces import (
    eval_actors, face_vote_label, pending_clips, pick_frames, read_summary,
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
