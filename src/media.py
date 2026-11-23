"""Bounded CPU-only FFmpeg rendering and FFprobe validation."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def find_tool(name: str, configured=None) -> str:
    local_path = Path(__file__).resolve().parents[1] / "configs/local.json"
    local = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
    candidate = configured or os.environ.get(name.upper() + "_BINARY") or local.get(name) or shutil.which(name)
    if candidate:
        resolved = shutil.which(str(candidate)) or str(Path(candidate).expanduser().resolve())
        if Path(resolved).is_file():
            return resolved
    raise RuntimeError(f"{name} missing. Install FFmpeg and FFprobe with: winget install --id Gyan.FFmpeg -e\n"
                       f"Reopen PowerShell or pass --{name} C:/path/to/{name}.exe")


def run(command, timeout=300):
    try:
        result = subprocess.run([str(c) for c in command], capture_output=True, text=True,
                                errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{Path(str(command[0])).name} exceeded {timeout} seconds") from exc
    if result.returncode:
        raise RuntimeError(f"{Path(str(command[0])).name} failed:\n{result.stderr[-4000:]}")
    return result.stdout


def probe(ffprobe: str, path: Path):
    return json.loads(run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", path], 30))


def render_scene(ffmpeg: str, image: Path, audio: Path, path: Path, duration: float, size, fps=24, threads=2):
    w, h = size
    frames = math.ceil(duration * fps)
    duration = frames / fps
    # One frame per input image; zoom is deliberately small to preserve caption margins.
    filters = (f"zoompan=z='min(zoom+0.00006,1.025)':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={w}x{h}:fps={fps},"
               f"fade=t=in:st=0:d=0.18,fade=t=out:st={max(0, duration - .18):.6f}:d=0.18,format=yuv420p")
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-filter_threads", "1",
         "-loop", "1", "-framerate", fps, "-i", image, "-i", audio,
         "-map", "0:v:0", "-map", "1:a:0", "-vf", filters, "-af", "apad",
         "-t", f"{duration:.6f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
         "-threads", threads, "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", path])
    return duration


def concatenate(ffmpeg: str, clips: list[Path], path: Path):
    # Relative, generated ASCII filenames avoid Windows drive and quote escaping in concat files.
    listing = path.parent / f"{path.stem}_concat.txt"
    listing.write_text("".join(f"file '{clip.name}'\n" for clip in clips), encoding="utf-8")
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1",
         "-i", listing, "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-movflags", "+faststart", path])


def validate_video(ffprobe: str, path: Path, size, expected_duration=None):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing or empty video: {path}")
    info = probe(ffprobe, path)
    videos = [s for s in info["streams"] if s.get("codec_type") == "video"]
    audios = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    if len(videos) != 1 or (videos[0]["width"], videos[0]["height"]) != tuple(size):
        raise ValueError(f"Unexpected dimensions in {path}")
    duration = float(info["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0 or not audios:
        raise ValueError(f"Video must have positive duration and an audio track: {path}")
    if expected_duration is not None and abs(duration - expected_duration) > 0.75:
        raise ValueError(f"Unexpected duration {duration:.3f}; expected {expected_duration:.3f}: {path}")
    if float(audios[0].get("duration", 0)) < duration - 0.75:
        raise ValueError(f"Audio does not cover video: {path}")
    return {"path": str(path.resolve()), "duration": duration, "width": size[0], "height": size[1],
            "audio_codec": audios[0]["codec_name"], "video_codec": videos[0]["codec_name"], "bytes": path.stat().st_size}
