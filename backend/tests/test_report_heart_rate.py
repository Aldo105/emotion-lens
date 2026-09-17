"""
Tests para el ritmo cardíaco en los reportes y la agregación por segundo
del CSV.
"""
import csv
import io

import pytest

from backend.app.services.report_generator import (
    CSV_HEADER,
    _csv_rows,
    _heart_rate_series,
    _heart_rate_stats,
    generate_csv_string,
)

FPS = 20


def _timeline(duration_s=5, bpm_fn=lambda t: 72.0, emotion_fn=lambda t: "neutral"):
    """Timeline sintético al ritmo real del pipeline: un registro por frame."""
    records = []
    for i in range(duration_s * FPS):
        t = i / FPS
        aus = {"AU04_brow_lowerer": 0.1, "AU12_lip_corner": 0.2}
        bpm = bpm_fn(t)
        if bpm is not None:
            aus["hr_bpm"] = bpm
            aus["hr_confidence"] = 0.8
        records.append({
            "timestamp": t,
            "emotion": emotion_fn(t),
            "confidence": 0.9,
            "congruence_score": 80.0,
            "action_units": aus,
        })
    return records


def _parse(csv_text):
    return list(csv.reader(io.StringIO(csv_text)))


# ── Serie y estadísticas de pulso ─────────────────────────────────────


def test_series_only_takes_records_that_carry_a_reading():
    # El pipeline solo guarda hr_bpm mientras la señal es utilizable, así que
    # una sesión puede tener huecos o ninguna lectura.
    timeline = _timeline(duration_s=4, bpm_fn=lambda t: 72.0 if t < 2 else None)

    series = _heart_rate_series(timeline)

    assert len(series) == 2 * FPS
    assert all(bpm == 72.0 for _, bpm, _ in series)


def test_zero_and_malformed_readings_are_not_counted_as_measurements():
    timeline = _timeline(duration_s=1, bpm_fn=lambda t: 0.0)
    assert _heart_rate_series(timeline) == []

    timeline[0]["action_units"]["hr_bpm"] = "sin señal"
    assert _heart_rate_series(timeline) == []


def test_stats_report_average_and_typical_range():
    # Sube linealmente de 60 a 80 a lo largo de la sesión.
    timeline = _timeline(duration_s=10, bpm_fn=lambda t: 60.0 + 2.0 * t)

    stats = _heart_rate_stats(timeline)

    assert stats["mean"] == pytest.approx(70.0, abs=0.5)
    assert stats["min"] == pytest.approx(60.0, abs=0.5)
    assert stats["max"] == pytest.approx(80.0, abs=0.5)
    assert stats["p25"] == pytest.approx(65.0, abs=0.5)
    assert stats["p75"] == pytest.approx(75.0, abs=0.5)
    assert stats["coverage"] == pytest.approx(1.0)


def test_typical_range_ignores_a_single_bad_window():
    # Una ventana ruidosa dispara el máximo; el rango intercuartílico no debe
    # moverse por ella, que es justo el motivo de reportarlo.
    timeline = _timeline(duration_s=5, bpm_fn=lambda t: 148.0 if t < 0.05 else 72.0)

    stats = _heart_rate_stats(timeline)

    assert stats["max"] == pytest.approx(148.0)
    assert stats["p75"] == pytest.approx(72.0)


def test_partial_coverage_is_reported():
    timeline = _timeline(duration_s=10, bpm_fn=lambda t: 72.0 if t < 3 else None)

    stats = _heart_rate_stats(timeline)

    assert stats["coverage"] == pytest.approx(0.3, abs=0.01)


def test_no_readings_yields_no_stats():
    assert _heart_rate_stats(_timeline(duration_s=3, bpm_fn=lambda t: None)) is None
    assert _heart_rate_stats([]) is None


# ── CSV ───────────────────────────────────────────────────────────────


def test_csv_emits_one_row_per_second_instead_of_one_per_frame():
    # 20 registros por segundo hacían que cada segundo se repitiera veinte
    # veces y unos minutos de entrevista fueran miles de filas casi iguales.
    timeline = _timeline(duration_s=6)

    rows = _csv_rows({"emotion_timeline": timeline})

    assert len(rows) == 6
    assert [r[0] for r in rows] == ["00:00", "00:01", "00:02", "00:03", "00:04", "00:05"]


def test_csv_time_column_is_readable_and_keeps_raw_seconds():
    timeline = _timeline(duration_s=1)
    timeline += [{**r, "timestamp": r["timestamp"] + 125} for r in _timeline(duration_s=1)]

    rows = _csv_rows({"emotion_timeline": timeline})

    assert rows[0][0] == "00:00" and rows[0][1] == 0
    assert rows[-1][0] == "02:05" and rows[-1][1] == 125


def test_csv_carries_the_heart_rate_for_that_second():
    timeline = _timeline(duration_s=3, bpm_fn=lambda t: 70.0 + t)

    rows = _csv_rows({"emotion_timeline": timeline})

    bpm_index = CSV_HEADER.index("bpm")
    assert float(rows[0][bpm_index]) == pytest.approx(70.5, abs=0.2)
    assert float(rows[2][bpm_index]) == pytest.approx(72.5, abs=0.2)


def test_csv_leaves_the_pulse_blank_for_seconds_without_a_reading():
    timeline = _timeline(duration_s=4, bpm_fn=lambda t: 72.0 if t < 2 else None)

    rows = _csv_rows({"emotion_timeline": timeline})
    bpm_index = CSV_HEADER.index("bpm")

    assert rows[0][bpm_index] != ""
    assert rows[3][bpm_index] == ""


def test_csv_second_reports_the_dominant_emotion_and_frame_count():
    # 15 frames de "happy" y 5 de "neutral" en el mismo segundo.
    timeline = _timeline(
        duration_s=1, emotion_fn=lambda t: "happy" if t < 0.75 else "neutral"
    )

    rows = _csv_rows({"emotion_timeline": timeline})

    assert rows[0][CSV_HEADER.index("emocion")] == "happy"
    assert rows[0][CSV_HEADER.index("frames")] == FPS


def test_csv_keeps_action_units_but_not_the_heart_rate_keys():
    # El pulso tiene su propia columna; repetirlo dentro del blob de AUs
    # obliga a parsear una cadena para leer un número que ya está al lado.
    rows = _csv_rows({"emotion_timeline": _timeline(duration_s=1)})

    aus = rows[0][CSV_HEADER.index("action_units")]
    assert "AU04_brow_lowerer=" in aus
    assert "hr_bpm" not in aus


def test_csv_string_has_the_header_and_one_row_per_second():
    text = generate_csv_string({"emotion_timeline": _timeline(duration_s=3)})

    parsed = _parse(text)
    assert parsed[0] == CSV_HEADER
    assert len(parsed) == 4


def test_empty_session_still_produces_a_valid_header():
    parsed = _parse(generate_csv_string({"emotion_timeline": []}))

    assert parsed == [CSV_HEADER]
