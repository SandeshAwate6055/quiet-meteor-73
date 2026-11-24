"""Provider contracts/adapters around the working Phase 1/2 implementations.

An image generator is deliberately request-only here. Future remote providers
must return candidates through the existing import/review/approval workflow.
"""
import hashlib
import os
from pathlib import Path
import shutil
from typing import Protocol

from . import media
from .narration import NarrationResult, synthesize
from .schema import Scene
from .image_providers import SceneImageProvider


def source_revision(*names):
    root = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in names)).hexdigest()


class StoryPlanner(Protocol):
    cache_key: str
    def plan(self, story, bible, settings, images, pipeline) -> tuple[dict, object]: ...


class ImageResolver(SceneImageProvider, Protocol):
    def resolve(self, scene: Scene) -> dict: ...


class ImageGenerationRequestProvider(Protocol):
    cache_key: str
    def prepare(self, story, bible, missing_ids: list[str], destination: Path) -> dict: ...


class VoiceProvider(Protocol):
    cache_key: str
    def synthesize(self, scene: Scene, directory: Path, settings: dict) -> NarrationResult: ...


class AnimationProvider(Protocol):
    cache_key: str
    def animate(self, image: Path, audio: Path, destination: Path, duration: float, size: tuple[int, int], settings: dict) -> float: ...


class FinalVideoProvider(Protocol):
    cache_key: str
    def assemble(self, clips: list[Path], destination: Path, size: tuple[int, int], duration: float) -> dict: ...
    def validate(self, path: Path, size: tuple[int, int], duration: float) -> dict: ...


class OfflineVoice:
    cache_key = source_revision("narration.py", "speech_worker.py", "sapi_narrate.ps1", "production_providers.py")

    def synthesize(self, scene, directory, settings):
        return synthesize(scene, directory, settings["rate"], settings["silent"])


class FFmpegAnimation:
    def __init__(self, ffmpeg):
        self.ffmpeg = ffmpeg
        self.cache_key = source_revision("media.py", "production_providers.py") + hashlib.sha256(media.run([ffmpeg, "-version"], 15).encode()).hexdigest()

    def animate(self, image, audio, destination, duration, size, settings, motion="push_in"):
        # Camera motion only. Character motion needs a separate image-to-video provider (see docs/ANIMATION_PLAN.md).
        return media.render_scene(self.ffmpeg, image, audio, destination, duration, size, settings["fps"], settings["threads"], motion)


class HybridAnimation:
    """Use an approved I2V MP4 clip when present for a shot, otherwise fall back to FFmpeg camera motion."""

    def __init__(self, ffmpeg, store):
        self.ffmpeg = ffmpeg
        self.store = store
        self.fallback = FFmpegAnimation(ffmpeg)
        self.cache_key = (
            source_revision("media.py", "production_providers.py", "asset_store.py")
            + hashlib.sha256(media.run([ffmpeg, "-version"], 15).encode()).hexdigest()
        )

    def clip_cache_key(self, scene_id: str) -> str:
        approved = self.store.approved("clip", scene_id)
        return approved[0]["sha256"] if approved else "camera_only"

    def animate(self, image, audio, destination, duration, size, settings, motion="push_in", scene_id=None):
        if scene_id:
            approved = self.store.approved("clip", scene_id)
            if approved:
                _, clip_path = approved
                return media.render_i2v_clip(
                    self.ffmpeg, clip_path, audio, destination, duration, size, settings["fps"], settings["threads"]
                )
        return self.fallback.animate(image, audio, destination, duration, size, settings, motion)


class FFmpegFinalVideo:
    def __init__(self, ffmpeg, ffprobe):
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe
        self.cache_key = source_revision("media.py", "production_providers.py") + hashlib.sha256((media.run([ffmpeg, "-version"], 15) + media.run([ffprobe, "-version"], 15)).encode()).hexdigest()

    def validate(self, path, size, duration):
        result = media.validate_video(self.ffprobe, path, size, duration)
        media.run([self.ffmpeg, "-v", "error", "-xerror", "-threads", "2", "-i", path, "-f", "null", "-"], 300)
        return {**result, "full_decode_passed": True}

    def assemble(self, clips, destination, size, duration):
        # Existing concat expects clips beside its listing. Stage files without
        # changing the renderer; hard links avoid extra copies when supported.
        staged = []
        for index, clip in enumerate(clips):
            target = destination.parent / f"part_{index:03d}.mp4"
            try:
                os.link(clip, target)
            except OSError:
                shutil.copyfile(clip, target)
            staged.append(target)
        media.concatenate(self.ffmpeg, staged, destination)
        return self.validate(destination, size, duration)
