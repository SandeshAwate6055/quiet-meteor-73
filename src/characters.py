"""Reusable character identity and prompt data. Reference paths may be absent initially."""
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath

from .schema import _identifier, _text

PROMPT_REVISION = "single_scene_v2"
REFERENCE_SPEC_REVISION = "single_view_v2"
SCENE_NEGATIVE_PROMPT = (
    "split panels, colored panels, collage, triptych, character sheet, lineup, multiple views, "
    "extra characters, duplicate animals, turquoise rabbit, text, watermark, pedestal"
)
REFERENCE_NEGATIVE_PROMPT = (
    "multiple views, front and back views, turnaround, character sheet, split panels, collage, "
    "extra characters, duplicate animal, pedestal, display stand, text, watermark"
)


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
    # Subject identities and action precede style, instead of being lost after a
    # long style/environment prefix. The backend rejects tokenizer truncation.
    ordered = sorted(scene.characters)
    count = {0: "zero", 1: "one", 2: "two"}.get(len(ordered), str(len(ordered)))
    subjects = []
    for n, identity in enumerate(ordered):
        character = bible.characters[identity]
        side = (" on the left" if n == 0 else " on the right") if len(ordered) == 2 else ""
        subjects.append(f"one {character.name.lower()} with {character.appearance}{side}")
    environment = "" if bible.environment.split()[-1].lower() in scene.visual_description.lower() else f"{bible.environment}. "
    return (f"Exactly {count} animals: {'; '.join(subjects) if subjects else 'empty landscape'}. "
            f"{scene.visual_description} {environment}One unified cinematic frame. {bible.style}.")


def reference_prompt(character: Character, bible: CharacterBible) -> str:
    pose = character.metadata.get("reference_pose", "single full-body three-quarter view, standing on plain ground")
    return f"Exactly one {character.name.lower()} with {character.appearance}. {pose}. {bible.style}, neutral background."
