"""
Make rendered videos playable in the browser.

The pip build of OpenCV used on the VPS ships without an H.264 encoder, so
cv2.VideoWriter falls back to 'mp4v' (MPEG-4 Part 2), which Chrome and most
browsers refuse to play. When that happens and a system ffmpeg is installed,
the file is re-encoded to H.264 in place. The local conda OpenCV writes
'avc1' directly, so there this is a no-op.
"""

import os
import shutil
import subprocess

# systemd runs the service with PATH limited to the venv, so look in the usual
# system location too.
_FFMPEG = shutil.which("ffmpeg") or ("/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else None)


def ensure_browser_playable(path: str, codec: str) -> None:
    """Re-encode `path` to H.264 when it was written with a codec browsers
    cannot play. Leaves the original untouched if ffmpeg is missing or fails."""
    if codec in ("avc1", "H264"):
        return
    if _FFMPEG is None:
        print(f"[WARN] {path} was written as '{codec}' and ffmpeg is not installed; "
              "browsers may not play it")
        return

    tmp_path = path + ".h264.mp4"
    result = subprocess.run(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", path,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(tmp_path):
        print(f"[WARN] ffmpeg could not re-encode {path} to H.264: {result.stderr.strip()[:200]}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return
    os.replace(tmp_path, path)
