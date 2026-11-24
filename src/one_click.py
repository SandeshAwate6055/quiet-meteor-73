"""One-click character-reference and challenge-scene handoff, extraction, execution and ZIP export.

Reuses the shared BatchWorker state machine and validate_result checks without
duplicating generation or validation logic in the notebook.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import uuid
import zipfile

from .asset_store import AssetStore, digest, read_image
from .characters import load_bible
from .image_batch import BatchSettings, BatchWorker, contact_sheet, inside, reference_sources, scene_seed, validate_request
from .image_conditioning import CONDITIONING_REVISION, reference_layout
from .modern_images import FluxKleinBackend, validate_profile
from .production_state import file_hash
from .schema import load_story

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_NAME = "character_references_v4_one_click"
DEFAULT_CHALLENGE_PACKAGE_NAME = "challenge_scene_v4_one_click"
DEFAULT_REMAINING_SCENES_PACKAGE_NAME = "remaining_scenes_v4_one_click"
DEFAULT_EPISODE_PACKAGE_NAME = "episode_01_scenes_v4_one_click"
DEFAULT_EPISODE_CORRECTIONS_PACKAGE_NAME = "episode_01_corrections_v4_one_click"
REMAINING_SCENE_IDS = ("racing_ahead", "nap", "steady_steps", "finish")
EPISODE_SHARED_STYLE = (
    "A polished friendly 3D children's cartoon render with rounded appealing shapes, soft matte materials, "
    "warm sunny outdoor forest lighting and natural ground shadows. One unified wide cinematic 16:9 frame, no text, no watermark."
)
EPISODE_NEGATIVE_COMMON = (
    "extra animals, duplicate rabbit, duplicate tortoise, multiple views, split panels, colored panels, "
    "collage, triptych, character sheet, lineup, bow tie, shorts, trousers, green shell, olive shell, gray skin, "
    "pedestal, text, title, letters, numbers, watermark, logo"
)


def validate_one_click_result(manifest_path, expected_request_hash=None):
    """Self-contained result validator that works both locally and inside extracted Colab bundles."""
    path = Path(manifest_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Result manifest must be an object")
    request_path = path.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    characters, scenes = validate_request(request)
    reference_only = bool(request.get("generation_policy", {}).get("reference_only"))
    profile_hash = request.get("generation_config", {}).get("backend_profile_sha256")
    if not profile_hash:
        raise ValueError("One-click request requires backend_profile_sha256")
    profile_path = path.parent / "generation_profile.json"
    if file_hash(profile_path) != profile_hash:
        raise ValueError("Result generation profile checksum mismatch")
    profile = validate_profile(json.loads(profile_path.read_text(encoding="utf-8")))
    actual_request_hash = file_hash(request_path)
    if data.get("version") != 1 or not isinstance(data.get("assets"), list) or data.get("request_sha256") != actual_request_hash:
        raise ValueError("Invalid result version/assets or request checksum")
    if expected_request_hash and actual_request_hash != expected_request_hash:
        raise ValueError("Result belongs to a different production request")
    seen = set()
    completed_scenes = []
    for row in data["assets"]:
        if not isinstance(row, dict):
            raise ValueError("One-click result assets must be objects")
        kind, identity = row.get("kind"), row.get("target_id")
        if (kind, identity) in seen or identity not in (characters if kind == "character" else scenes if kind == "scene" else {}):
            raise ValueError(f"Duplicate or unknown target: {kind}/{identity}")
        seen.add((kind, identity))
        image_path = inside(path.parent, row["file"])
        if file_hash(image_path) != row["sha256"]:
            raise ValueError(f"Result image checksum mismatch: {kind}/{identity}")
        image = read_image(image_path)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Result metadata must be an object")
        if not metadata.get("reused_reference"):
            if (metadata.get("model_id") != profile["model_id"] or
                    metadata.get("model_revision") != profile["model_revision"] or
                    metadata.get("model_license") != profile["model_license"]):
                raise ValueError("Result model provenance differs from its pinned generation profile")
            target_spec = (characters if kind == "character" else scenes)[identity]
            if metadata.get("prompt") != target_spec["prompt"] or list(image.size) != metadata.get("resolution"):
                raise ValueError("Result prompt or resolution differs from its request/decoded image")
            expected_negative = target_spec.get("negative_prompt", request["negative_prompt"])
            if metadata.get("negative_prompt", expected_negative) != expected_negative:
                raise ValueError("Result negative prompt differs from its request")
            if metadata.get("seed") != scene_seed(identity, data["settings"]["base_seed"]):
                raise ValueError("Generated seed does not match the deterministic batch seed")
        if kind == "scene":
            identities = scenes[identity]["characters"]
            expected_cond = CONDITIONING_REVISION if identities else "not_applicable"
            if metadata.get("conditioning") != expected_cond:
                raise ValueError("Scene lacks required reference conditioning")
            if identities and metadata.get("conditioning_layout") != reference_layout(identities):
                raise ValueError("Scene reference placement does not match the declared conditioning layout")
            refs = metadata.get("references", {})
            if set(refs) != set(identities):
                raise ValueError("Scene reference metadata does not match character IDs")
            for ref in refs.values():
                ref_path = inside(path.parent, ref["file"])
                if file_hash(ref_path) != ref["sha256"]:
                    raise ValueError("Scene conditioning reference checksum mismatch")
                read_image(ref_path)
            if data.get("jobs", {}).get(f"scene:{identity}", {}).get("status") != "completed":
                raise ValueError("Scene image has no completed job record")
            completed_scenes.append(identity)
    missing_scenes = [s for s in scenes if s not in completed_scenes]
    failed_scenes = [s for s in scenes if data.get("jobs", {}).get(f"scene:{s}", {}).get("status") == "failed"]
    expected_completion = {
        "completed_scene_ids": [s for s in scenes if s in completed_scenes],
        "missing_scene_ids": missing_scenes,
        "failed_scene_ids": failed_scenes,
    }
    if reference_only:
        if data.get("requested_reference_ids") != list(characters):
            raise ValueError("Requested reference IDs differ from the reference-only request")
        completed_refs = [c for c in characters if ("character", c) in seen]
        missing_refs = [c for c in characters if c not in completed_refs]
        failed_refs = [c for c in characters if data.get("jobs", {}).get(f"character:{c}", {}).get("status") == "failed"]
        expected_completion.update(
            completed_reference_ids=completed_refs,
            missing_reference_ids=missing_refs,
            failed_reference_ids=failed_refs,
        )
        for row in data["assets"]:
            identity, metadata = row["target_id"], row["metadata"]
            if (data.get("jobs", {}).get(f"character:{identity}", {}).get("status") not in ("completed", "reused") or
                    metadata.get("reference_spec_revision") != characters[identity].get("reference_spec_revision") or
                    metadata.get("conditioning") != "text_only"):
                raise ValueError("Reference-only result lacks a valid job/specification/text-only record")
        if data.get("status") == "complete" and (missing_refs or failed_refs):
            raise ValueError("Batch claims complete while required references are missing/failed")
    if data.get("completion") != expected_completion:
        raise ValueError("Result completion summary does not match its actual images/jobs")
    if data.get("status") == "complete" and (missing_scenes or failed_scenes):
        raise ValueError("Batch claims complete while requested scenes are missing/failed")
    result = {
        "manifest_path": str(path),
        "manifest_sha256": file_hash(path),
        "request_sha256": actual_request_hash,
        "status": data.get("status"),
        "asset_count": len(data["assets"]),
        "completed_scene_ids": expected_completion["completed_scene_ids"],
        "missing_scene_ids": missing_scenes,
        "failed_scene_ids": failed_scenes,
        "errors": data.get("errors", []),
        "all_checksums_match": True,
        "all_images_decode": True,
    }
    if reference_only:
        result.update({
            key: expected_completion[key]
            for key in ("completed_reference_ids", "missing_reference_ids", "failed_reference_ids")
        })
    return result


def load_profile(profile_path=None):
    path = Path(profile_path or (ROOT / "configs/one_click_generation.json")).resolve()
    payload = path.read_bytes()
    profile = validate_profile(json.loads(payload.decode("utf-8")))
    return profile, payload, digest(payload)


def load_challenge_profile(profile_path=None):
    return load_profile(profile_path or (ROOT / "configs/one_click_challenge_scene.json"))


def load_remaining_scenes_profile(profile_path=None):
    return load_profile(profile_path or (ROOT / "configs/one_click_remaining_scenes.json"))


def build_one_click_request(story, bible, profile, profile_sha256, *, characters=("hare", "tortoise"), attempt=1):
    validate_profile(profile)
    characters = list(characters)
    if not characters or len(set(characters)) != len(characters) or set(characters) - {"hare", "tortoise"}:
        raise ValueError("Select hare, tortoise, or both exactly once")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 1000:
        raise ValueError("attempt must be an integer from 1 to 1000")
    from .assets import export_request
    with tempfile.TemporaryDirectory(prefix="one_click_req_") as temp:
        directory = Path(temp)
        store = AssetStore(directory / "empty_references")
        request_path = directory / "request.json"
        export_request(store, story, bible, request_path, ["challenge"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
    rows = {c["id"]: c for c in request["characters"]}
    spec_rev = profile["revision"]
    shared_style = profile["shared_style"].strip()
    selected_rows = []
    for identity in characters:
        row = dict(rows[identity])
        char_spec = profile["characters"][identity]
        subject = char_spec["subject"].strip()
        neg = char_spec["negative_prompt"].strip()
        meta = dict(row.get("metadata", {}))
        meta.update(
            reference_subject=subject,
            reference_negative_prompt=neg,
            reference_spec_revision=spec_rev,
        )
        row.update(
            metadata=meta,
            prompt=f"{subject} {shared_style}",
            negative_prompt=neg,
            reference_spec_revision=spec_rev,
        )
        selected_rows.append(row)
    base_seed = profile["settings"]["base_seed"] + (attempt - 1)
    request["characters"] = selected_rows
    request["scenes"] = []
    request["locked_scene_ids"] = []
    request["environment"] = "Plain warm-gray studio background"
    request["reference_prompt_revision"] = spec_rev
    request["generation_config"].update(
        family=profile["settings"]["family"],
        steps=profile["settings"]["steps"],
        reference_size=profile["settings"]["reference_size"],
        base_seed=base_seed,
        backend_profile_sha256=profile_sha256,
    )
    request["generation_policy"] = {
        "reference_only": True,
        "required_reference_ids": characters,
        "allowed_scene_ids": [],
        "full_batch_enabled": False,
        "fresh_references": True,
        "reference_spec_revision": spec_rev,
        "reference_attempt": attempt,
    }
    request["quality_review"] = {
        "previous_references": "v2_and_v3_failed_human_visual_review",
        "approved": False,
    }
    return request


def build_one_click_challenge_request(store, story, bible, profile, profile_sha256, *, attempt=1):
    validate_profile(profile)
    if profile.get("mode") != "challenge_scene":
        raise ValueError("Expected a challenge_scene generation profile")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 1000:
        raise ValueError("attempt must be an integer from 1 to 1000")
    for identity in ("hare", "tortoise"):
        if not store.approved("character", identity):
            raise ValueError(f"Character reference '{identity}' must be approved in AssetStore before preparing challenge scene")
    from .assets import export_request
    with tempfile.TemporaryDirectory(prefix="one_click_challenge_req_") as temp:
        directory = Path(temp)
        request_path = directory / "request.json"
        export_request(store, story, bible, request_path, ["challenge"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
    for row in request["characters"]:
        if "approved_reference_png_base64" not in row or "approved_reference_sha256" not in row:
            raise ValueError(f"Missing embedded approved reference in request for character: {row.get('id')}")
    if [s["id"] for s in request["scenes"]] != ["challenge"]:
        raise ValueError("Challenge request must contain only the 'challenge' scene")
    if "challenge" in request.get("locked_scene_ids", []):
        raise ValueError("Scene 'challenge' is currently locked by an approved image; unlock it before generating a new candidate")
    scene_spec = profile["scenes"]["challenge"]
    subject = scene_spec["subject"].strip()
    neg = scene_spec["negative_prompt"].strip()
    shared_style = profile["shared_style"].strip()
    request["scenes"][0].update(
        prompt=f"{subject} {shared_style}",
        negative_prompt=neg,
        scene_spec_revision=profile["revision"],
    )
    base_seed = profile["settings"]["base_seed"] + (attempt - 1)
    request["generation_config"].update(
        family=profile["settings"]["family"],
        steps=profile["settings"]["steps"],
        reference_size=profile["settings"]["reference_size"],
        base_seed=base_seed,
        backend_profile_sha256=profile_sha256,
    )
    request["generation_policy"] = {
        "reference_only": False,
        "allowed_scene_ids": ["challenge"],
        "full_batch_enabled": False,
        "require_reference_review": False,
        "scene_spec_revision": profile["revision"],
        "scene_attempt": attempt,
    }
    request["quality_review"] = {
        "references": "v4_flux_klein_approved_by_human_visual_review",
        "approved": False,
    }
    return request


def build_one_click_remaining_scenes_request(store, story, bible, profile, profile_sha256, *, attempt=1):
    validate_profile(profile)
    if profile.get("mode") != "remaining_scenes":
        raise ValueError("Expected a remaining_scenes generation profile")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 1000:
        raise ValueError("attempt must be an integer from 1 to 1000")
    for identity in ("hare", "tortoise"):
        if not store.approved("character", identity):
            raise ValueError(f"Character reference '{identity}' must be approved in AssetStore before preparing remaining scenes")
    approved_challenge = store.approved("scene", "challenge")
    if not approved_challenge:
        raise ValueError("Scene 'challenge' must be approved and locked in AssetStore before preparing the remaining four scenes")
    challenge_record, _ = approved_challenge
    from .assets import export_request
    with tempfile.TemporaryDirectory(prefix="one_click_remaining_req_") as temp:
        directory = Path(temp)
        request_path = directory / "request.json"
        export_request(store, story, bible, request_path, list(REMAINING_SCENE_IDS))
        request = json.loads(request_path.read_text(encoding="utf-8"))
    for row in request["characters"]:
        if "approved_reference_png_base64" not in row or "approved_reference_sha256" not in row:
            raise ValueError(f"Missing embedded approved reference in request for character: {row.get('id')}")
    if tuple(s["id"] for s in request["scenes"]) != REMAINING_SCENE_IDS:
        raise ValueError(f"Remaining-scenes request must contain exactly {list(REMAINING_SCENE_IDS)}")
    if "challenge" in {s["id"] for s in request["scenes"]}:
        raise ValueError("Approved scene 'challenge' must never be scheduled in the remaining-scenes request")
    if request.get("locked_scene_ids"):
        raise ValueError(f"One or more remaining scenes are already locked by an approval: {request['locked_scene_ids']}")
    shared_style = profile["shared_style"].strip()
    for scene_row in request["scenes"]:
        sid = scene_row["id"]
        scene_spec = profile["scenes"][sid]
        subject = scene_spec["subject"].strip()
        neg = scene_spec["negative_prompt"].strip()
        scene_row.update(
            prompt=f"{subject} {shared_style}",
            negative_prompt=neg,
            scene_spec_revision=profile["revision"],
        )
    base_seed = profile["settings"]["base_seed"] + (attempt - 1)
    request["generation_config"].update(
        family=profile["settings"]["family"],
        steps=profile["settings"]["steps"],
        reference_size=profile["settings"]["reference_size"],
        base_seed=base_seed,
        backend_profile_sha256=profile_sha256,
    )
    request["generation_policy"] = {
        "reference_only": False,
        "allowed_scene_ids": list(REMAINING_SCENE_IDS),
        "excluded_approved_scene_ids": ["challenge"],
        "reviewed_smoke_scene_id": "challenge",
        "reviewed_smoke_asset_id": challenge_record["asset_id"],
        "reviewed_smoke_sha256": challenge_record["sha256"],
        "full_batch_enabled": True,
        "require_reference_review": False,
        "scene_spec_revision": profile["revision"],
        "scene_attempt": attempt,
    }
    request["quality_review"] = {
        "references": "v4_flux_klein_approved_by_human_visual_review",
        "challenge_scene": "v4_flux_klein_approved_and_locked_in_asset_store",
        "approved": False,
    }
    return request


def build_one_click_episode_bundle(store, story, bible, base_profile, *, attempt=1):
    """Build an episode_scenes profile and request for missing shots only, reusing approved scenes."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 1000:
        raise ValueError("attempt must be an integer from 1 to 1000")
    for identity in ("hare", "tortoise"):
        if not store.approved("character", identity):
            raise ValueError(f"Character reference '{identity}' must be approved in AssetStore before preparing episode scenes")
    approved_scenes = {}
    for sc in story.scenes:
        target = sc.reuse_asset or sc.id
        found = store.approved("scene", target)
        if found:
            approved_scenes[sc.id] = (target, found[0])
    # Require at least one reviewed/approved scene (e.g. challenge) as quality gate
    smoke_ref = store.approved("scene", "challenge") or next((v[1] for v in approved_scenes.values()), None)
    if not smoke_ref:
        raise ValueError("At least one approved scene (e.g. 'challenge') is required before preparing an episode batch")
    smoke_record = smoke_ref[0] if isinstance(smoke_ref, tuple) else smoke_ref

    missing_scenes = [sc for sc in story.scenes if sc.id not in approved_scenes]
    if not missing_scenes:
        raise ValueError("All story scenes already have approved assets; no new image generation needed")

    profile = json.loads(json.dumps(base_profile))
    profile["mode"] = "episode_scenes"
    profile["revision"] = base_profile.get("revision") or "episode_scenes_flux_klein_v4"
    profile["max_new_images"] = len(missing_scenes)
    profile["settings"]["base_seed"] = base_profile["settings"]["base_seed"] + (attempt - 1)
    profile["shared_style"] = base_profile.get("shared_style") or EPISODE_SHARED_STYLE
    existing_specs = base_profile.get("scenes") or {}
    profile["scenes"] = {}
    for sc in missing_scenes:
        if sc.id in existing_specs:
            profile["scenes"][sc.id] = {
                "subject": existing_specs[sc.id]["subject"].strip(),
                "negative_prompt": existing_specs[sc.id]["negative_prompt"].strip(),
            }
            continue
        chars = set(sc.characters)
        extra_neg = []
        if chars == {"hare"}:
            extra_neg.append("tortoise, turtle, second animal")
        elif chars == {"tortoise"}:
            extra_neg.append("rabbit, hare, second animal, clothing")
        neg = (", ".join(extra_neg) + ", " if extra_neg else "") + EPISODE_NEGATIVE_COMMON
        profile["scenes"][sc.id] = {
            "subject": sc.visual_description.strip(),
            "negative_prompt": neg,
        }
    validate_profile(profile)
    profile_bytes = json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    profile_sha256 = digest(profile_bytes)

    from .assets import export_request
    with tempfile.TemporaryDirectory(prefix="one_click_ep_req_") as temp:
        request_path = Path(temp) / "request.json"
        export_request(store, story, bible, request_path, [sc.id for sc in missing_scenes])
        request = json.loads(request_path.read_text(encoding="utf-8"))
    for row in request["characters"]:
        if "approved_reference_png_base64" not in row or "approved_reference_sha256" not in row:
            raise ValueError(f"Missing embedded approved reference in request for character: {row.get('id')}")
    if request.get("locked_scene_ids"):
        raise ValueError(f"Missing scenes unexpectedly locked: {request['locked_scene_ids']}")

    shared_style = profile["shared_style"].strip()
    for scene_row in request["scenes"]:
        sid = scene_row["id"]
        spec = profile["scenes"][sid]
        scene_row.update(
            prompt=f"{spec['subject']} {shared_style}",
            negative_prompt=spec["negative_prompt"],
            scene_spec_revision=profile["revision"],
        )

    request["generation_config"].update(
        family=profile["settings"]["family"],
        steps=profile["settings"]["steps"],
        reference_size=profile["settings"]["reference_size"],
        base_seed=profile["settings"]["base_seed"],
        backend_profile_sha256=profile_sha256,
    )
    request["generation_policy"] = {
        "reference_only": False,
        "episode_scene_ids": [sc.id for sc in missing_scenes],
        "allowed_scene_ids": [sc.id for sc in missing_scenes],
        "excluded_approved_scene_ids": sorted({t for t, _ in approved_scenes.values()}),
        "reused_shot_ids": [sc.id for sc in story.scenes if sc.id in approved_scenes],
        "reviewed_smoke_scene_id": smoke_record["target_id"],
        "reviewed_smoke_asset_id": smoke_record["asset_id"],
        "reviewed_smoke_sha256": smoke_record["sha256"],
        "full_batch_enabled": True,
        "require_reference_review": False,
        "scene_spec_revision": profile["revision"],
        "scene_attempt": attempt,
    }
    request["quality_review"] = {
        "references": "v4_flux_klein_approved_by_human_visual_review",
        "approved_scenes_preserved": sorted({t for t, _ in approved_scenes.values()}),
        "approved": False,
    }
    return request, profile, profile_bytes, profile_sha256, approved_scenes


def prepare_one_click_episode_handoff(script_path=None, destination=None, *, name=DEFAULT_EPISODE_PACKAGE_NAME,
                                      asset_dir=None, attempt=1, profile_path=None, overwrite=False):
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple alphanumeric file name")
    destination = Path(destination or (ROOT / "examples")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    store = AssetStore(asset_dir or (ROOT / "assets/library"))
    base_profile, _, _ = load_challenge_profile(profile_path)
    story = load_story(Path(script_path or (ROOT / "scripts/episode_01_golden_acorn.json")))
    bible = load_bible(ROOT / "configs/character_bible.json")
    request, profile, profile_bytes, profile_sha256, approved_scenes = build_one_click_episode_bundle(
        store, story, bible, base_profile, attempt=attempt
    )
    req_txt = (ROOT / "notebooks/requirements-one-click.txt").read_bytes()
    from .batch_handoff import create_handoff
    with tempfile.TemporaryDirectory(prefix="one_click_ep_pkg_") as temp:
        directory = Path(temp)
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False),
                                encoding="utf-8", newline="\n")
        package_path = directory / "package.zip"
        metadata = create_handoff(
            request_path,
            store,
            package_path,
            extra_files={
                "generation_profile.json": profile_bytes,
                "requirements-one-click.txt": req_txt,
            },
        )
        copies = ((request_path, destination / f"{name}.json"), (package_path, destination / f"{name}.zip"))
        if not overwrite:
            for source, target in copies:
                if target.exists() and file_hash(target) != file_hash(source):
                    raise ValueError(f"Existing artifact differs: {target}. Choose a NEW --name; nothing was overwritten.")
        for source, target in copies:
            if overwrite or not target.exists():
                target.write_bytes(source.read_bytes())
    return {
        **metadata,
        "package_path": str(destination / f"{name}.zip"),
        "request_path": str(destination / f"{name}.json"),
        "profile_sha256": profile_sha256,
        "approved_reference_ids": ["hare", "tortoise"],
        "reused_shot_ids": [sc.id for sc in story.scenes if sc.id in approved_scenes],
        "excluded_approved_scene_ids": sorted({t for t, _ in approved_scenes.values()}),
        "scene_ids": [s["id"] for s in request["scenes"]],
        "base_seed": request["generation_config"]["base_seed"],
        "generation_started": False,
    }


def prepare_one_click_handoff(destination=None, *, name=DEFAULT_PACKAGE_NAME, characters=("hare", "tortoise"),
                              attempt=1, profile_path=None, overwrite=False):
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple alphanumeric file name")
    destination = Path(destination or (ROOT / "examples")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    profile, profile_bytes, profile_sha256 = load_profile(profile_path)
    if attempt != 1:
        profile = json.loads(profile_bytes.decode("utf-8"))
        profile["settings"]["base_seed"] = profile["settings"]["base_seed"] + (attempt - 1)
        profile_bytes = json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        profile_sha256 = digest(profile_bytes)
    story = load_story(ROOT / "scripts/tortoise_and_hare.json")
    bible = load_bible(ROOT / "configs/character_bible.json")
    request = build_one_click_request(story, bible, profile, profile_sha256, characters=characters, attempt=attempt)
    req_txt = (ROOT / "notebooks/requirements-one-click.txt").read_bytes()
    from .batch_handoff import create_handoff
    with tempfile.TemporaryDirectory(prefix="one_click_pkg_") as temp:
        directory = Path(temp)
        store = AssetStore(directory / "empty_references")
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False),
                                encoding="utf-8", newline="\n")
        package_path = directory / "package.zip"
        metadata = create_handoff(
            request_path,
            store,
            package_path,
            extra_files={
                "generation_profile.json": profile_bytes,
                "requirements-one-click.txt": req_txt,
            },
        )
        copies = ((request_path, destination / f"{name}.json"), (package_path, destination / f"{name}.zip"))
        if not overwrite:
            for source, target in copies:
                if target.exists() and file_hash(target) != file_hash(source):
                    raise ValueError(f"Existing artifact differs: {target}. Choose a NEW --name; nothing was overwritten.")
        for source, target in copies:
            if overwrite or not target.exists():
                target.write_bytes(source.read_bytes())
    return {
        **metadata,
        "package_path": str(destination / f"{name}.zip"),
        "request_path": str(destination / f"{name}.json"),
        "profile_sha256": profile_sha256,
        "reference_ids": list(characters),
        "scene_ids": [],
        "base_seed": request["generation_config"]["base_seed"],
        "generation_started": False,
    }


def prepare_one_click_challenge_handoff(destination=None, *, name=DEFAULT_CHALLENGE_PACKAGE_NAME,
                                        asset_dir=None, attempt=1, profile_path=None, overwrite=False):
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple alphanumeric file name")
    destination = Path(destination or (ROOT / "examples")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    store = AssetStore(asset_dir or (ROOT / "assets/library"))
    profile, profile_bytes, profile_sha256 = load_challenge_profile(profile_path)
    if attempt != 1:
        profile = json.loads(profile_bytes.decode("utf-8"))
        profile["settings"]["base_seed"] = profile["settings"]["base_seed"] + (attempt - 1)
        profile_bytes = json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        profile_sha256 = digest(profile_bytes)
    story = load_story(ROOT / "scripts/tortoise_and_hare.json")
    bible = load_bible(ROOT / "configs/character_bible.json")
    request = build_one_click_challenge_request(store, story, bible, profile, profile_sha256, attempt=attempt)
    req_txt = (ROOT / "notebooks/requirements-one-click.txt").read_bytes()
    from .batch_handoff import create_handoff
    with tempfile.TemporaryDirectory(prefix="one_click_challenge_pkg_") as temp:
        directory = Path(temp)
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False),
                                encoding="utf-8", newline="\n")
        package_path = directory / "package.zip"
        metadata = create_handoff(
            request_path,
            store,
            package_path,
            extra_files={
                "generation_profile.json": profile_bytes,
                "requirements-one-click.txt": req_txt,
            },
        )
        copies = ((request_path, destination / f"{name}.json"), (package_path, destination / f"{name}.zip"))
        if not overwrite:
            for source, target in copies:
                if target.exists() and file_hash(target) != file_hash(source):
                    raise ValueError(f"Existing artifact differs: {target}. Choose a NEW --name; nothing was overwritten.")
        for source, target in copies:
            if overwrite or not target.exists():
                target.write_bytes(source.read_bytes())
    return {
        **metadata,
        "package_path": str(destination / f"{name}.zip"),
        "request_path": str(destination / f"{name}.json"),
        "profile_sha256": profile_sha256,
        "approved_reference_ids": ["hare", "tortoise"],
        "scene_ids": ["challenge"],
        "base_seed": request["generation_config"]["base_seed"],
        "generation_started": False,
    }


def prepare_one_click_remaining_scenes_handoff(destination=None, *, name=DEFAULT_REMAINING_SCENES_PACKAGE_NAME,
                                               asset_dir=None, attempt=1, profile_path=None, overwrite=False):
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple alphanumeric file name")
    destination = Path(destination or (ROOT / "examples")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    store = AssetStore(asset_dir or (ROOT / "assets/library"))
    profile, profile_bytes, profile_sha256 = load_remaining_scenes_profile(profile_path)
    if attempt != 1:
        profile = json.loads(profile_bytes.decode("utf-8"))
        profile["settings"]["base_seed"] = profile["settings"]["base_seed"] + (attempt - 1)
        profile_bytes = json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        profile_sha256 = digest(profile_bytes)
    story = load_story(ROOT / "scripts/tortoise_and_hare.json")
    bible = load_bible(ROOT / "configs/character_bible.json")
    request = build_one_click_remaining_scenes_request(store, story, bible, profile, profile_sha256, attempt=attempt)
    req_txt = (ROOT / "notebooks/requirements-one-click.txt").read_bytes()
    from .batch_handoff import create_handoff
    with tempfile.TemporaryDirectory(prefix="one_click_remaining_pkg_") as temp:
        directory = Path(temp)
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False),
                                encoding="utf-8", newline="\n")
        package_path = directory / "package.zip"
        metadata = create_handoff(
            request_path,
            store,
            package_path,
            extra_files={
                "generation_profile.json": profile_bytes,
                "requirements-one-click.txt": req_txt,
            },
        )
        copies = ((request_path, destination / f"{name}.json"), (package_path, destination / f"{name}.zip"))
        if not overwrite:
            for source, target in copies:
                if target.exists() and file_hash(target) != file_hash(source):
                    raise ValueError(f"Existing artifact differs: {target}. Choose a NEW --name; nothing was overwritten.")
        for source, target in copies:
            if overwrite or not target.exists():
                target.write_bytes(source.read_bytes())
    return {
        **metadata,
        "package_path": str(destination / f"{name}.zip"),
        "request_path": str(destination / f"{name}.json"),
        "profile_sha256": profile_sha256,
        "approved_reference_ids": ["hare", "tortoise"],
        "excluded_approved_scene_ids": ["challenge"],
        "scene_ids": list(REMAINING_SCENE_IDS),
        "base_seed": request["generation_config"]["base_seed"],
        "generation_started": False,
    }


def extract_and_verify_bundle(package_path, extract_root, *, expected_package_sha256=None):
    package_path = Path(package_path).resolve()
    payload = package_path.read_bytes()
    package_hash = digest(payload)
    if expected_package_sha256 and package_hash != expected_package_sha256:
        raise ValueError(f"Handoff package SHA256 mismatch: expected {expected_package_sha256}, got {package_hash}")
    bundle = Path(extract_root).resolve() / package_hash
    with zipfile.ZipFile(package_path) as archive:
        if sum(m.file_size for m in archive.infolist()) > 128 * 1024 * 1024 or len(archive.infolist()) > 200:
            raise ValueError("Unexpectedly large handoff archive")
        index = json.loads(archive.read("bundle_index.json"))
        expected = set(index["files"]) | {"bundle_index.json"}
        if set(archive.namelist()) != expected:
            raise ValueError("Unexpected handoff files in bundle")
        for required in ("request.json", "references.json", "generation_profile.json", "requirements-one-click.txt"):
            if required not in index["files"]:
                raise ValueError(f"One-click handoff missing required file: {required}")
        for name in archive.namelist():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe ZIP path in handoff")
            member_bytes = archive.read(name)
            if name != "bundle_index.json" and hashlib.sha256(member_bytes).hexdigest() != index["files"][name]:
                raise ValueError(f"Handoff file checksum mismatch: {name}")
            target = inside(bundle, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != member_bytes:
                    raise ValueError(f"Existing extracted bundle file was modified: {name}")
            else:
                with target.open("xb") as stream:
                    stream.write(member_bytes)
    request_path = bundle / "request.json"
    if file_hash(request_path) != index["request_sha256"]:
        raise ValueError("Extracted request.json checksum mismatch")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    profile_path = bundle / "generation_profile.json"
    profile = validate_profile(json.loads(profile_path.read_text(encoding="utf-8")))
    expected_profile_hash = request.get("generation_config", {}).get("backend_profile_sha256")
    if not expected_profile_hash or file_hash(profile_path) != expected_profile_hash:
        raise ValueError("Extracted generation_profile.json checksum does not match request")
    return {
        "bundle_dir": bundle,
        "package_sha256": package_hash,
        "request_sha256": index["request_sha256"],
        "profile_sha256": expected_profile_hash,
        "request": request,
        "profile": profile,
    }


def run_one_click_batch(bundle_dir, work_root, *, backend_factory=None, cache_dir=None, result_batch_id=None,
                        allow_partial=False):
    bundle_dir = Path(bundle_dir).resolve()
    work_root = Path(work_root).resolve()
    request_path = bundle_dir / "request.json"
    profile_path = bundle_dir / "generation_profile.json"
    references_path = bundle_dir / "references.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    profile = validate_profile(json.loads(profile_path.read_text(encoding="utf-8")))
    if file_hash(profile_path) != request.get("generation_config", {}).get("backend_profile_sha256"):
        raise ValueError("Generation profile checksum mismatch before execution")
    profile_mode = profile.get("mode", "references")
    cfg = profile["settings"]
    settings = BatchSettings(
        family=cfg["family"],
        steps=cfg["steps"],
        base_seed=request["generation_config"]["base_seed"],
        reference_size=cfg["reference_size"],
    )
    references = reference_sources(references_path) if references_path.exists() else []
    probe = BatchWorker(request_path, work_root / "_probe", references=references, settings=settings)
    if profile_mode == "references":
        if request.get("generation_policy", {}).get("reference_only") is not True or request.get("scenes") != []:
            raise ValueError("One-click reference execution only allows zero-scene character reference requests")
        default_prefix = "references_v4_"
        worker_mode = "references"
        smoke_scene = None
        max_budget = min(profile["max_new_images"], len(request["characters"]))
        quality_reviewed = False
    elif profile_mode == "challenge_scene":
        if (request.get("generation_policy", {}).get("reference_only") is True or
                [s["id"] for s in request.get("scenes", [])] != ["challenge"]):
            raise ValueError("One-click challenge execution only allows the single 'challenge' scene request")
        default_prefix = "challenge_scene_v4_"
        worker_mode = "smoke"
        smoke_scene = "challenge"
        max_budget = 1
        quality_reviewed = False
    elif profile_mode == "remaining_scenes":
        if (request.get("generation_policy", {}).get("reference_only") is True or
                tuple(s["id"] for s in request.get("scenes", [])) != REMAINING_SCENE_IDS or
                "challenge" in {s["id"] for s in request.get("scenes", [])} or
                not request.get("generation_policy", {}).get("reviewed_smoke_sha256")):
            raise ValueError("One-click remaining_scenes execution requires approved 'challenge' smoke and exactly the 4 remaining scenes")
        default_prefix = "remaining_scenes_v4_"
        worker_mode = "full"
        smoke_scene = REMAINING_SCENE_IDS[0]
        max_budget = min(profile["max_new_images"], len(REMAINING_SCENE_IDS))
        quality_reviewed = True
    elif profile_mode == "episode_scenes":
        policy = request.get("generation_policy", {})
        ids = [s["id"] for s in request.get("scenes", [])]
        if (policy.get("reference_only") is True or not ids or set(ids) != set(profile["scenes"]) or
                policy.get("episode_scene_ids") != ids or not policy.get("reviewed_smoke_sha256")):
            raise ValueError("Episode execution requires an approved smoke scene and a request matching exactly its profile's missing shots")
        if set(ids) & set(policy.get("excluded_approved_scene_ids", [])):
            raise ValueError("Approved scenes must never be regenerated by an episode batch")
        default_prefix = "episode_scenes_v4_"
        worker_mode = "full"
        smoke_scene = ids[0]
        max_budget = min(profile["max_new_images"], len(ids))
        quality_reviewed = True
    else:
        raise ValueError(f"Unsupported one-click profile mode: {profile_mode}")
    batch_name = result_batch_id or f"{default_prefix}{probe.generation_identity[:16]}"
    if not batch_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple alphanumeric result batch name")
    result_dir = work_root / "results" / batch_name
    worker = BatchWorker(request_path, result_dir, references=references, settings=settings)
    preflight = worker.preflight(mode=worker_mode, smoke_scene=smoke_scene, max_new_images=max_budget,
                                 quality_reviewed=quality_reviewed)
    if preflight["blockers"]:
        raise ValueError("Preflight blocked: " + "; ".join(preflight["blockers"]))
    if profile_mode in ("challenge_scene", "remaining_scenes", "episode_scenes"):
        if any(task["kind"] == "character" for task in preflight["new_images"]):
            raise ValueError("Scene execution refused to regenerate character references; approved references must be reused")
        if any(r.get("approval") != "approved" for r in preflight["reused_images"] if r.get("kind") == "character"):
            raise ValueError("Scene execution requires APPROVED character references")
        if profile_mode == "remaining_scenes" and any(task.get("id") == "challenge" for task in preflight["new_images"]):
            raise ValueError("Approved scene 'challenge' must never be regenerated")
    result_dir.mkdir(parents=True, exist_ok=True)
    profile_snapshot = result_dir / "generation_profile.json"
    profile_bytes = profile_path.read_bytes()
    if profile_snapshot.exists():
        if profile_snapshot.read_bytes() != profile_bytes:
            raise ValueError("Persistent result generation_profile.json was modified")
    else:
        with profile_snapshot.open("xb") as stream:
            stream.write(profile_bytes)
    cache_dir = Path(cache_dir or (work_root / "model_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if backend_factory is None:
        def backend_factory(worker_settings, previous):
            return FluxKleinBackend(worker_settings, profile, cache_dir, previous)
    outcome = worker.execute(backend_factory, mode=worker_mode, smoke_scene=smoke_scene,
                             max_new_images=max_budget, quality_reviewed=quality_reviewed)
    sheet_path = contact_sheet(result_dir)
    validation = validate_one_click_result(worker.manifest_path, expected_request_hash=worker.request_hash)
    export_dir = work_root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    archive_base = export_dir / f"{result_dir.name}_{uuid.uuid4().hex[:12]}"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=result_dir))
    if outcome.get("status") != "complete" and not allow_partial:
        raise RuntimeError(
            f"Batch did not complete ({outcome.get('status')}); partial export saved at {archive_path}: {outcome.get('errors')}"
        )
    return {
        "result_dir": result_dir,
        "manifest_path": worker.manifest_path,
        "contact_sheet_path": sheet_path,
        "archive_path": archive_path,
        "archive_sha256": file_hash(archive_path),
        "preflight": {k: v for k, v in preflight.items() if k != "chosen_references"},
        "validation": validation,
        "outcome": outcome,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("references", "challenge", "remaining", "episode"), default="references")
    parser.add_argument("--script", type=Path, default=ROOT / "scripts/episode_01_golden_acorn.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples")
    parser.add_argument("--name")
    parser.add_argument("--characters", nargs="+", choices=("hare", "tortoise"), default=["hare", "tortoise"])
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "challenge":
        print(json.dumps(prepare_one_click_challenge_handoff(
            args.output_dir, name=args.name or DEFAULT_CHALLENGE_PACKAGE_NAME, attempt=args.attempt
        ), indent=2))
    elif args.mode == "remaining":
        print(json.dumps(prepare_one_click_remaining_scenes_handoff(
            args.output_dir, name=args.name or DEFAULT_REMAINING_SCENES_PACKAGE_NAME, attempt=args.attempt
        ), indent=2))
    elif args.mode == "episode":
        print(json.dumps(prepare_one_click_episode_handoff(
            args.script, args.output_dir, name=args.name or DEFAULT_EPISODE_PACKAGE_NAME, attempt=args.attempt
        ), indent=2))
    else:
        print(json.dumps(prepare_one_click_handoff(
            args.output_dir, name=args.name or DEFAULT_PACKAGE_NAME, characters=args.characters, attempt=args.attempt
        ), indent=2))
