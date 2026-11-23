"""Small dependency-free validator; keep aligned with configs/scene_schema.json."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class Scene:
    id: str
    narration: str
    duration: float
    visual_description: str
    characters: tuple[str, ...] = ()


@dataclass(frozen=True)
class Story:
    title: str
    scenes: tuple[Scene, ...]


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
    if not isinstance(data, dict) or set(data) != {"title", "scenes"}:
        raise ValueError("Story requires exactly title and scenes")
    title = _text(data["title"], "title", 100)
    raw_scenes = data["scenes"]
    if not isinstance(raw_scenes, list) or not 1 <= len(raw_scenes) <= 30:
        raise ValueError("Story must contain 1 to 30 scenes")
    scenes, seen = [], set()
    required = {"id", "narration", "duration", "visual_description"}
    for raw in raw_scenes:
        if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - {"characters"}:
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
        scenes.append(Scene(scene_id, _text(raw["narration"], "narration", 1000), float(duration),
                            _text(raw["visual_description"], "visual_description", 500), characters))
    return Story(title, tuple(scenes))
