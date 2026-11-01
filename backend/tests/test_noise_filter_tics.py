from backend.app.services.noise_filter import FacialNoiseFilter

REST = {"AU1": 1.0, "AU4": 0.0, "AU5": 1.0, "AU12": 0.4, "AU23": 0.0, "AU25": 0.05,
        "AU26": 1.0, "AU45": 0.0}


def _run(frame_at, seconds=12.0, fps=20):
    f = FacialNoiseFilter()
    flagged = 0
    n = int(seconds * fps)
    for i in range(n):
        t = i / fps
        st = f.analyze({**REST, **frame_at(t)}, t)
        flagged += st.is_tic
    return flagged / n


def _pulse(t, every, width=0.15):
    return (t % every) < width


def test_a_repeated_brow_twitch_is_still_a_tic():
    # AU4 jumping every 1.5 s with the rest of the face still.
    assert _run(lambda t: {"AU4": 0.9 if _pulse(t, 1.5) else 0.0}) > 0.3


def test_normal_blinking_is_not_a_tic():
    # Several blinks every 10 s is ordinary; AU5 moves with them.
    assert _run(lambda t: {"AU45": 1.0, "AU5": 0.2} if _pulse(t, 2.5) else {}) == 0


def test_talking_mouth_is_not_a_tic():
    # Lips opening and closing with lip tension spikes, as in speech.
    def speech(t):
        open_ = _pulse(t, 0.4, 0.2)
        return {"AU25": 1.0 if open_ else 0.0, "AU23": 0.6 if _pulse(t, 1.2) else 0.0}
    assert _run(speech) == 0


def test_resting_aus_near_one_do_not_count_without_calibration():
    # AU1/AU26 sit at 1.0 at rest in this pipeline; small dips are noise.
    assert _run(lambda t: {"AU1": 0.75 if _pulse(t, 2.0) else 1.0}) == 0
