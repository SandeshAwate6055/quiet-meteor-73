"""Prepare fresh, reference-only v3 requests using existing export/handoff APIs.

No images or models are loaded. A single-character retry is a new request with
a new deterministic seed; it never regenerates the other character.
"""
import argparse
import json
from pathlib import Path
import tempfile

from src.asset_store import AssetStore
from src.assets import export_request
from src.batch_handoff import create_handoff
from src.characters import load_bible, REFERENCE_SPEC_REVISION
from src.production_state import file_hash
from src.schema import load_story

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination, name="character_references_v3", characters=("hare", "tortoise"), attempt=1):
    characters = list(characters)
    if not characters or len(set(characters)) != len(characters) or set(characters) - {"hare", "tortoise"}:
        raise ValueError("Select hare, tortoise, or both exactly once")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 1000:
        raise ValueError("attempt must be an integer from 1 to 1000")
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple file name")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bible = load_bible(ROOT / "configs/character_bible.json")
    story = load_story(ROOT / "scripts/tortoise_and_hare.json")
    with tempfile.TemporaryDirectory(prefix="reference_request_") as temp:
        directory = Path(temp)
        # Historical candidates/approval locks remain untouched in the real library.
        store = AssetStore(directory / "empty_references")
        request_path = directory / "request.json"
        export_request(store, story, bible, request_path, ["challenge"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        rows = {c["id"]: c for c in request["characters"]}
        request["characters"] = [rows[c] for c in characters]
        request["scenes"] = []
        request["locked_scene_ids"] = []
        request["environment"] = "Plain neutral gray studio background"
        request["reference_prompt_revision"] = REFERENCE_SPEC_REVISION
        request["generation_config"].update(reference_size=1024, base_seed=20261123 + attempt)
        request["generation_policy"] = {
            "reference_only": True, "required_reference_ids": characters,
            "allowed_scene_ids": [], "full_batch_enabled": False, "fresh_references": True,
            "reference_spec_revision": REFERENCE_SPEC_REVISION, "reference_attempt": attempt,
        }
        request["quality_review"] = {"previous_references": "v2_failed_human_visual_review", "approved": False}
        # Canonical LF bytes keep the tracked JSON and packaged request identical
        # across Windows/Linux checkouts (the files are new temporary inputs).
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False),
                                encoding="utf-8", newline="\n")
        package_path = directory / "package.zip"
        metadata = create_handoff(request_path, store, package_path)
        copies = ((request_path, destination / f"{name}.json"), (package_path, destination / f"{name}.zip"))
        for source, target in copies:
            if target.exists() and file_hash(target) != file_hash(source):
                raise ValueError(f"Existing artifact differs: {target}. Choose a NEW --name; nothing was overwritten.")
        for source, target in copies:
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(source.read_bytes())
    return {**metadata, "package_path": str(destination / f"{name}.zip"),
            "request_path": str(destination / f"{name}.json"), "reference_ids": characters,
            "scene_ids": [], "base_seed": 20261123 + attempt, "generation_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples")
    parser.add_argument("--name", default="character_references_v3")
    parser.add_argument("--characters", nargs="+", choices=("hare", "tortoise"), default=["hare", "tortoise"])
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.name, args.characters, args.attempt), indent=2))
