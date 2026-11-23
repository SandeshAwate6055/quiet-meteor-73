"""Reusable character identity and prompt data. Reference paths may be absent initially."""
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath

from .schema import _identifier, _text


@dataclass(frozen=True)
class Character:
    id: str
    name: str
    appearance: str
    reference_image_paths: tuple[str, ...]
    metadata: dict


@dataclass(frozen=True)
class CharacterBible:
    style: str
    environment: str
    characters: dict[str, Character]
    metadata: dict


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Asset paths must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Asset paths cannot escape their root")
    return value


def load_bible(path: Path) -> CharacterBible:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("characters"), list):
        raise ValueError("Character Bible requires version 1 and a characters list")
    characters = {}
    for item in data["characters"]:
        if not isinstance(item, dict):
            raise ValueError("Character must be an object")
        identity = _identifier(item.get("id"))
        if identity in characters:
            raise ValueError(f"Duplicate character id: {identity}")
        paths = item.get("reference_image_paths", [])
        if not isinstance(paths, list) or not isinstance(item.get("metadata", {}), dict):
            raise ValueError("Invalid reference paths or character metadata")
        characters[identity] = Character(identity, _text(item.get("name"), "name", 100),
            _text(item.get("appearance"), "appearance", 3000), tuple(safe_relative(p) for p in paths), item.get("metadata", {}))
    if not characters or not isinstance(data.get("metadata", {}), dict):
        raise ValueError("Character Bible needs characters and object metadata")
    return CharacterBible(_text(data.get("style"), "style", 3000),
                          _text(data.get("environment"), "environment", 3000), characters, data.get("metadata", {}))


def scene_prompt(scene, bible: CharacterBible) -> str:
    unknown = set(scene.characters) - set(bible.characters)
    if unknown:
        raise ValueError(f"Characters absent from Bible: {sorted(unknown)}")
    identities = "; ".join(f"{bible.characters[c].name}: {bible.characters[c].appearance}" for c in scene.characters)
    return f"{bible.style}. {bible.environment}. {scene.visual_description} Characters: {identities}. Wide story illustration, clear silhouettes, full bodies, no lettering."
