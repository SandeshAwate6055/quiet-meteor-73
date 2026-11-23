"""Deterministic structured-story planning; no text generation or translation."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from .characters import scene_prompt

SIZES = {"landscape": (1280, 720), "vertical": (720, 1280)}
DEFAULTS = {"target_duration": None, "language": "en-US", "style": None,
            "formats": ["landscape", "vertical"], "vertical_scene_ids": None,
            "narration": {"rate": 0, "silent": False}, "animation": "zoom_pan"}


def load_settings(path: Path | None):
    supplied = json.loads(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown production settings; see configs/production.json")
    settings = {**DEFAULTS, **supplied}
    voice = supplied.get("narration", {})
    if not isinstance(voice, dict) or set(voice) - {"rate", "silent"}:
        raise ValueError("Narration settings support only rate and silent")
    settings["narration"] = {**DEFAULTS["narration"], **voice}
    rate = settings["narration"]["rate"]
    if isinstance(rate, bool) or not isinstance(rate, int) or not -10 <= rate <= 10 or not isinstance(settings["narration"]["silent"], bool):
        raise ValueError("Narration rate must be an integer -10..10; silent must be boolean")
    duration = settings["target_duration"]
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or not 0 < duration <= 3600):
        raise ValueError("target_duration must be null or 0 < seconds <= 3600")
    if settings["language"] not in ("en", "en-US", "en-GB"):
        raise ValueError("Current offline voice adapter supports English scripts only (en/en-US/en-GB); translation and other voices need a future provider")
    if settings["animation"] != "zoom_pan":
        raise ValueError("Only zoom_pan animation is implemented")
    formats = settings["formats"]
    if not isinstance(formats, list) or not formats or any(not isinstance(f, str) or f not in SIZES for f in formats) or len(set(formats)) != len(formats):
        raise ValueError("formats must be a unique nonempty list of landscape/vertical")
    style = settings["style"]
    if style is not None and (not isinstance(style, str) or not style.strip() or len(style) > 3000):
        raise ValueError("style must be null or nonempty text, at most 3000 characters")
    selected = settings["vertical_scene_ids"]
    if selected is not None and (not isinstance(selected, list) or not selected or any(not isinstance(s, str) for s in selected) or len(set(selected)) != len(selected)):
        raise ValueError("vertical_scene_ids must be null or a unique nonempty list of scene ids")
    return settings


class StructuredStoryPlanner:
    cache_key = "structured_story_v1"

    def plan(self, story, bible, settings, images, pipeline):
        effective_bible = replace(bible, style=settings["style"]) if settings["style"] else bible
        ids = {s.id for s in story.scenes}
        selected = settings["vertical_scene_ids"]
        if selected is None:
            defaults = pipeline.get("vertical_scene_ids", [])
            if not isinstance(defaults, list) or any(not isinstance(value, str) for value in defaults) or len(set(defaults)) != len(defaults):
                raise ValueError("Pipeline vertical_scene_ids must be a list of unique scene ids")
            selected = defaults if defaults and set(defaults) <= ids else [s.id for s in story.scenes[-3:]]
        if set(selected) - ids:
            raise ValueError(f"Unknown vertical scene ids: {sorted(set(selected) - ids)}")
        scenes = []
        for index, scene in enumerate(story.scenes, 1):
            prompt = scene_prompt(scene, effective_bible)  # Validates all character ids.
            formats = [f for f in settings["formats"] if f != "vertical" or scene.id in selected]
            selection = images.resolve(scene)
            scenes.append({**asdict(scene), "index": index, "characters": list(scene.characters),
                "formats": formats, "image_selection": selection, "image_prompt": prompt,
                "required_assets": [{"kind": "scene", "id": scene.id},
                    *({"kind": "character", "id": identity} for identity in scene.characters)],
                "status": "pending" if formats else "not_requested", "narration": {}, "images": {}, "clips": {}})
        requested = sum(s.duration for s in story.scenes)
        target = settings["target_duration"] or requested
        warnings = []
        if abs(requested - target) > max(5, target * .2):
            warnings.append(f"Script scene durations total {requested:.2f}s versus target {target:.2f}s; target is advisory and the story will not be stretched.")
        if settings["style"] and settings["style"] != bible.style:
            warnings.append("Style override is used for future image prompts; existing approved images and Pillow cards are not automatically restyled.")
        return {"planner": self.cache_key, "title": story.title, "settings": settings,
                "character_bible": asdict(effective_bible), "requested_scene_seconds": requested,
                "target_duration": target, "vertical_scene_ids": selected, "scenes": scenes,
                "warnings": warnings}, effective_bible
