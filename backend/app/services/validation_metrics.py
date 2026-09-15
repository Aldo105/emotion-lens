"""
EmotionLens — Human Validation Metrics

Turns the post-session survey's per-moment verdicts into measured agreement
between what the system flagged and what a human reviewer actually confirmed.

This is what makes the tool auditable rather than a black box: instead of
asking anyone to trust a relevance score, the system reports how often its
own flags survive human review, broken down by emotion, by relevance band,
and by whether the flag contradicted the displayed emotion.

Verdict values come from the survey UI: "correct" | "incorrect" | "unsure",
or None when the reviewer skipped that moment.
"""

CONFIRMED = "correct"
REJECTED = "incorrect"
UNSURE = "unsure"

# Relevance bands — the engine's own minimum threshold is 60 (see
# settings.micro_expr_relevance_threshold), so "low" should normally be empty.
HIGH_RELEVANCE_MIN = 80
MEDIUM_RELEVANCE_MIN = 60


def _empty_bucket() -> dict:
    return {"confirmed": 0, "rejected": 0, "unsure": 0, "agreement_rate": None}


def _tally(bucket: dict, verdict: str) -> None:
    if verdict == CONFIRMED:
        bucket["confirmed"] += 1
    elif verdict == REJECTED:
        bucket["rejected"] += 1
    elif verdict == UNSURE:
        bucket["unsure"] += 1


def _finalize(bucket: dict) -> dict:
    """Agreement rate ignores 'unsure' — it's confirmed vs explicitly rejected."""
    decided = bucket["confirmed"] + bucket["rejected"]
    bucket["agreement_rate"] = (bucket["confirmed"] / decided) if decided else None
    return bucket


def _relevance_band(score) -> str:
    if score is None:
        return "unknown"
    if score >= HIGH_RELEVANCE_MIN:
        return "high"
    if score >= MEDIUM_RELEVANCE_MIN:
        return "medium"
    return "low"


def compute_agreement_metrics(validations: list[dict] | None) -> dict:
    """
    Compute agreement metrics over a flat list of moment-validation records.

    Each record looks like:
        {"timestamp": float, "emotion": str, "verdict": str|None,
         "relevance_score": int|None, "is_contradictory": bool|None}

    Records the reviewer never answered (verdict None) are counted as
    "unanswered" and excluded from every rate, so skipping the survey can't
    inflate or deflate the numbers.

    Returns None rates (never 0.0) when there's nothing decided, so callers
    can distinguish "no data yet" from "the system was always wrong".
    """
    overall = _empty_bucket()
    unanswered = 0
    by_emotion: dict[str, dict] = {}
    by_relevance: dict[str, dict] = {}
    by_contradiction: dict[str, dict] = {}

    for record in validations or []:
        verdict = record.get("verdict")
        if verdict not in (CONFIRMED, REJECTED, UNSURE):
            unanswered += 1
            continue

        _tally(overall, verdict)

        emotion = record.get("emotion") or "unknown"
        _tally(by_emotion.setdefault(emotion, _empty_bucket()), verdict)

        band = _relevance_band(record.get("relevance_score"))
        _tally(by_relevance.setdefault(band, _empty_bucket()), verdict)

        is_contradictory = record.get("is_contradictory")
        if is_contradictory is not None:
            key = "contradictory" if is_contradictory else "congruent"
            _tally(by_contradiction.setdefault(key, _empty_bucket()), verdict)

    return {
        **_finalize(overall),
        "unanswered": unanswered,
        "total_moments": len(validations or []),
        "by_emotion": {k: _finalize(v) for k, v in by_emotion.items()},
        "by_relevance_band": {k: _finalize(v) for k, v in by_relevance.items()},
        "by_contradiction": {k: _finalize(v) for k, v in by_contradiction.items()},
    }


def aggregate_agreement_metrics(feedback_rows: list) -> dict:
    """
    Same metrics, pooled across many SessionFeedback rows — the global
    "how often does this system agree with humans" number.

    Also reports how many sessions actually contributed a decided verdict,
    since a single heavily-reviewed session shouldn't read as broad evidence.
    """
    flat: list[dict] = []
    sessions_with_validations = 0

    for row in feedback_rows:
        validations = getattr(row, "moment_validations", None) or []
        decided = [
            v for v in validations
            if v.get("verdict") in (CONFIRMED, REJECTED, UNSURE)
        ]
        if decided:
            sessions_with_validations += 1
        flat.extend(validations)

    metrics = compute_agreement_metrics(flat)
    metrics["sessions_with_validations"] = sessions_with_validations
    return metrics
