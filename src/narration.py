"""Replace synthesize() to plug in a future local or Colab speech provider."""
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

from .schema import Scene


@dataclass(frozen=True)
class NarrationResult:
    path: Path
    duration: float
    backend: str
    warnings: tuple[str, ...] = ()


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    if duration <= 0:
        raise ValueError(f"Empty WAV: {path}")
    return duration


def write_silence(path: Path, duration: float):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22050)
        # Chunk writes keep memory use bounded even for longer scenes.
        remaining = round(duration * 22050)
        while remaining:
            count = min(22050, remaining)
            audio.writeframes(b"\0\0" * count)
            remaining -= count


def synthesize(scene: Scene, directory: Path, rate: int = 0, force_silence: bool = False) -> NarrationResult:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scene.id}.wav"
    text_path = directory / f"{scene.id}.txt"
    text_path.write_text(scene.narration, encoding="utf-8")
    warnings = []
    commands = []
    if not force_silence:
        if os.name == "nt":
            ps = shutil.which("powershell.exe") or str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe")
            commands.append(("windows_sapi", [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                str(Path(__file__).with_name("sapi_narrate.ps1")), "-TextPath", str(text_path.resolve()),
                "-OutputPath", str(path.resolve()), "-Rate", str(rate)]))
        commands.append(("pyttsx3", [sys.executable, "-m", "src.speech_worker", str(text_path.resolve()), str(path.resolve())]))
    for backend, command in commands:
        try:
            path.unlink(missing_ok=True)
            subprocess.run(command, capture_output=True, text=True, errors="replace", check=True,
                           timeout=45, cwd=Path(__file__).resolve().parents[1])
            return NarrationResult(path, wav_duration(path), backend, tuple(warnings))
        except (OSError, subprocess.SubprocessError, ValueError, wave.Error, EOFError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            warnings.append(f"{backend} failed: {str(detail).strip()[-700:]}")
    write_silence(path, scene.duration)
    warnings.append("Speech unavailable or disabled: silent audio substituted; narration remains on scene cards.")
    return NarrationResult(path, wav_duration(path), "silence", tuple(warnings))
