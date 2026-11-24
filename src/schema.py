"""Small dependency-free validator; keep aligned with configs/scene_schema.json."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re


MOTIONS = ("push_in", "pull_out", "pan_left", "pan_right", "hold")  # Camera motion only; see docs/ANIMATION_PLAN.md
ANIMATION_INTENTS = ("camera_only", "character_motion_candidate")
STORY_OPTIONAL = {"logline", "moral", "target_duration", "shorts"}
SCENE_OPTIONAL = {"characters", "reuse_asset", "camera", "motion", "animation_intent", "continuity", "pause_after", "focus", "act"}


@dataclass(frozen=True)
class Scene:
    id: str
    narration: str
    duration: float  # Minimum hold; measured narration (+ pause_after) can extend it.
    visual_description: str
    characters: tuple[str, ...] = ()
    reuse_asset: str | None = None  # Approved scene asset target whose image this shot reuses.
    camera: str = ""
    motion: str = "push_in"
    animation_intent: str = "camera_only"
    continuity: dict = field(default_factory=dict)
    pause_after: float = 0.0
    focus: float = 0.5  # Horizontal subject centre (0-1) for subject-preserving vertical crops.
    act: str = ""


@dataclass(frozen=True)
class Story:
    title: str
    scenes: tuple[Scene, ...]
    meta: dict = field(default_factory=dict)  # Optional logline, moral, target_duration, shorts.


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text, at most {limit} characters")
    return value.strip()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", value):
        raise ValueError("Identifiers must match [a-z][a-z0-9_]{0,39}")
    return value


def load_story(path: Path) -> Story:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not {"title", "scenes"} <= set(data) or set(data) - {"title", "scenes"} - STORY_OPTIONAL:
        raise ValueError("Story requires title and scenes, plus only optional logline/moral/target_duration/shorts")
    title = _text(data["title"], "title", 100)
    raw_scenes = data["scenes"]
    if not isinstance(raw_scenes, list) or not 1 <= len(raw_scenes) <= 30:
        raise ValueError("Story must contain 1 to 30 scenes")
    scenes, seen = [], set()
    required = {"id", "narration", "duration", "visual_description"}
    for raw in raw_scenes:
        if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - SCENE_OPTIONAL:
            raise ValueError("Scene fields must follow configs/scene_schema.json")
        scene_id = _identifier(raw["id"])
        if scene_id in seen:
            raise ValueError(f"Duplicate scene id: {scene_id}")
        seen.add(scene_id)
        duration = raw["duration"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 120:
            raise ValueError(f"Invalid duration for {scene_id}: expected 0 < seconds <= 120")
        characters = raw.get("characters", [])
        if not isinstance(characters, list):
            raise ValueError("characters must be a list")
        characters = tuple(_identifier(c) for c in characters)
        if len(set(characters)) != len(characters):
            raise ValueError("characters must be unique")
        extra = _scene_extras(scene_id, raw)
        scenes.append(Scene(scene_id, _text(raw["narration"], "narration", 1000), float(duration),
                            _text(raw["visual_description"], "visual_description", 500), characters, **extra))
    meta = {key: data[key] for key in STORY_OPTIONAL if key in data}
    for key in ("logline", "moral"):
        if key in meta:
            _text(meta[key], key, 400)
    target = meta.get("target_duration")
    if target is not None and (isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or not 0 < target <= 3600):
        raise ValueError("target_duration must be 0 < seconds <= 3600")
    if "shorts" in meta and not isinstance(meta["shorts"], list):
        raise ValueError("shorts must be a list")
    return Story(title, tuple(scenes), meta)


def _scene_extras(scene_id, raw):
    extra = {}
    if "reuse_asset" in raw:
        extra["reuse_asset"] = _identifier(raw["reuse_asset"])
    if "camera" in raw:
        extra["camera"] = _text(raw["camera"], "camera", 200)
    if "act" in raw:
        extra["act"] = _text(raw["act"], "act", 60)
    if "motion" in raw:
        if raw["motion"] not in MOTIONS:
            raise ValueError(f"{scene_id}: motion must be one of {MOTIONS}")
        extra["motion"] = raw["motion"]
    if "animation_intent" in raw:
        if raw["animation_intent"] not in ANIMATION_INTENTS:
            raise ValueError(f"{scene_id}: animation_intent must be one of {ANIMATION_INTENTS}")
        extra["animation_intent"] = raw["animation_intent"]
    if "continuity" in raw:
        value = raw["continuity"]
        if (not isinstance(value, dict) or len(value) > 12 or
                any(not isinstance(k, str) or not isinstance(v, str) or len(v) > 160 for k, v in value.items())):
            raise ValueError(f"{scene_id}: continuity must be a small object of short text values")
        extra["continuity"] = dict(value)
    for key, low, high in (("pause_after", 0, 5), ("focus", 0, 1)):
        if key in raw:
            number = raw[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high:
                raise ValueError(f"{scene_id}: {key} must be between {low} and {high}")
            extra[key] = float(number)
    return extra
