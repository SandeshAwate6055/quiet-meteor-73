"""Vertical Shorts extraction from approved episode assets (distinct from full-episode vertical export)."""
import json
from pathlib import Path

from .asset_store import AssetStore
from .image_providers import FileImageProvider, compose_illustration
from .media import find_tool, validate_video
from .production_plan import SIZES
from .production_providers import HybridAnimation, FFmpegFinalVideo, OfflineVoice
from .production_state import file_hash, write_json
from .schema import Story, load_story


def plan_shorts(story: Story) -> list[dict]:
    """Validate and return declared Shorts edits (selected hook-to-ending moments)."""
    raw_shorts = story.meta.get("shorts", [])
    by_id = {s.id: s for s in story.scenes}
    planned = []
    seen = set()
    for item in raw_shorts:
        if not isinstance(item, dict) or not {"id", "title", "shots"} <= set(item):
            raise ValueError("Each Short requires id, title and shots")
        sid = item["id"]
        if sid in seen:
            raise ValueError(f"Duplicate Short id: {sid}")
        seen.add(sid)
        shots = item["shots"]
        if not isinstance(shots, list) or len(shots) < 2 or any(s not in by_id for s in shots):
            raise ValueError(f"Short '{sid}' must reference at least 2 valid scene IDs")
        est_duration = sum(by_id[s].duration + by_id[s].pause_after for s in shots)
        planned.append({
            "id": sid,
            "title": item["title"],
            "shots": list(shots),
            "shot_count": len(shots),
            "estimated_duration_seconds": round(est_duration, 2),
            "vertical_focus": [by_id[s].focus for s in shots],
        })
    return planned


def render_short(story_path: Path, short_id: str, output_dir: Path, *,
                 asset_dir: Path, pipeline_path: Path, ffmpeg=None, ffprobe=None, voice=None) -> dict:
    """Render one standalone 720x1280 vertical Short from selected story shots."""
    story = load_story(Path(story_path))
    shorts = {item["id"]: item for item in plan_shorts(story)}
    if short_id not in shorts:
        raise ValueError(f"Unknown Short id: {short_id}")
    spec = shorts[short_id]
    by_id = {s.id: s for s in story.scenes}
    pipeline = json.loads(Path(pipeline_path).read_text(encoding="utf-8"))
    ffmpeg = find_tool("ffmpeg", ffmpeg or pipeline.get("ffmpeg"))
    ffprobe = find_tool("ffprobe", ffprobe or pipeline.get("ffprobe"))
    store = AssetStore(asset_dir)
    images = FileImageProvider(store)
    voice = voice or OfflineVoice()
    animation = HybridAnimation(ffmpeg, store)
    renderer = FFmpegFinalVideo(ffmpeg, ffprobe)

    out = Path(output_dir).resolve()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    size = SIZES["vertical"]
    clips = []
    total = 0.0
    shot_records = []
    for idx, sid in enumerate(spec["shots"], 1):
        scene = by_id[sid]
        narration = voice.synthesize(scene, out / "audio", {"rate": pipeline.get("speech_rate", 0), "silent": False})
        duration = max(scene.duration, narration.duration + (pipeline["tail_seconds"] if narration.backend != "silence" else 0) + scene.pause_after)
        img_path = out / "images" / f"{sid}_vertical.png"
        selection = images.resolve(scene)
        if selection["provider"] == "approved_file":
            compose_illustration(Path(selection["source_path"]), scene, img_path, size)
        else:
            images.render(scene, story.title, idx, len(spec["shots"]), img_path, size)
        clip_path = out / "clips" / f"{idx:02d}_{sid}.mp4"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        actual = animation.animate(img_path, narration.path, clip_path, duration, size, pipeline, scene.motion, scene_id=sid)
        clips.append(clip_path)
        total += actual
        mode = "i2v_clip" if animation.clip_cache_key(sid) != "camera_only" else "camera_only"
        shot_records.append({"id": sid, "duration": actual, "provider": selection["provider"], "motion": scene.motion, "animation_mode": mode})

    final_path = out / f"{short_id}.mp4"
    validation = renderer.assemble(clips, final_path, size, total)
    manifest = {
        "kind": "story_studio_short",
        "short_id": short_id,
        "title": spec["title"],
        "story_title": story.title,
        "shots": shot_records,
        "output": {**validation, "path": str(final_path), "sha256": file_hash(final_path)},
    }
    write_json(out / f"{short_id}_manifest.json", manifest)
    return manifest
