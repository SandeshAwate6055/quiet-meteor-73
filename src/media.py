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


def motion_filter(motion: str, frames: int, w: int, h: int, fps: int) -> str:
    """Camera-only motion presets (no character animation). push_in is the historical default."""
    centre = "x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2'".replace("'", "'")
    presets = {
        "push_in": "z='min(zoom+0.00006,1.025)':" + centre,
        "pull_out": "z='if(lte(zoom,1.0),1.04,max(1.0,zoom-0.00006))':" + centre,
        "pan_right": f"z='1.06':x='(iw-iw/zoom)*on/{frames}':y='ih/2-ih/zoom/2'",
        "pan_left": f"z='1.06':x='(iw-iw/zoom)*(1-on/{frames})':y='ih/2-ih/zoom/2'",
        "hold": "z='1':" + centre,
    }
    if motion not in presets:
        raise ValueError(f"Unknown camera motion preset: {motion}")
    return f"zoompan={presets[motion]}:d=1:s={w}x{h}:fps={fps}"


def render_scene(ffmpeg: str, image: Path, audio: Path, path: Path, duration: float, size, fps=24, threads=2, motion="push_in"):
    w, h = size
    frames = math.ceil(duration * fps)
    duration = frames / fps
    # One frame per input image; motion is deliberately small to preserve caption margins.
    filters = (f"{motion_filter(motion, frames, w, h, fps)},"
               f"fade=t=in:st=0:d=0.18,fade=t=out:st={max(0, duration - .18):.6f}:d=0.18,format=yuv420p")
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-filter_threads", "1",
         "-loop", "1", "-framerate", fps, "-i", image, "-i", audio,
         "-map", "0:v:0", "-map", "1:a:0", "-vf", filters, "-af", "apad",
         "-t", f"{duration:.6f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
         "-threads", threads, "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", path])
    return duration


def render_i2v_clip(ffmpeg: str, clip_path: Path, audio: Path, path: Path, duration: float, size, fps=24, threads=2):
    """Fit an approved I2V MP4 clip to target dimensions and exact narration duration."""
    w, h = size
    frames = math.ceil(duration * fps)
    duration = frames / fps
    graph = (
        f"[0:v]fps={fps},split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=24:2[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgf];"
        f"[bgb][fgf]overlay=(W-w)/2:(H-h)/2,"
        f"fade=t=in:st=0:d=0.18,fade=t=out:st={max(0, duration - .18):.6f}:d=0.18,format=yuv420p[v]"
    )
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-filter_threads", "1",
         "-stream_loop", "-1", "-i", clip_path, "-i", audio,
         "-filter_complex", graph, "-map", "[v]", "-map", "1:a:0", "-af", "apad",
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


def mix_music(ffmpeg: str, video: Path, music: Path, path: Path, volume_db: float = -22.0, fade_out: float = 3.0, duration=None):
    """Mix licensed background music under existing narration; video stream is copied unchanged."""
    total = duration if duration is not None else float(json.loads(run([ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_format", "-of", "json", video], 30))["format"]["duration"])
    fade_start = max(0.0, total - fade_out)
    graph = (f"[1:a]volume={volume_db}dB,afade=t=in:st=0:d=1.5,afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}[m];"
             "[0:a][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-stream_loop", "-1", "-i", music,
         "-filter_complex", graph, "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
         "-ar", "48000", "-ac", "2", "-t", f"{total:.3f}", "-movflags", "+faststart", path], 600)


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
