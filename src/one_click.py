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


def run_one_click_batch(bundle_dir, work_root, *, backend_factory=None, cache_dir=None, result_batch_id=None):
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
        max_budget = min(profile["max_new_images"], len(request["characters"]))
    elif profile_mode == "challenge_scene":
        if (request.get("generation_policy", {}).get("reference_only") is True or
                [s["id"] for s in request.get("scenes", [])] != ["challenge"]):
            raise ValueError("One-click challenge execution only allows the single 'challenge' scene request")
        default_prefix = "challenge_scene_v4_"
        worker_mode = "smoke"
        max_budget = 1
    else:
        raise ValueError(f"Unsupported one-click profile mode: {profile_mode}")
    batch_name = result_batch_id or f"{default_prefix}{probe.generation_identity[:16]}"
    if not batch_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple alphanumeric result batch name")
    result_dir = work_root / "results" / batch_name
    worker = BatchWorker(request_path, result_dir, references=references, settings=settings)
    preflight = worker.preflight(mode=worker_mode, smoke_scene="challenge", max_new_images=max_budget)
    if preflight["blockers"]:
        raise ValueError("Preflight blocked: " + "; ".join(preflight["blockers"]))
    if profile_mode == "challenge_scene":
        if any(task["kind"] == "character" for task in preflight["new_images"]):
            raise ValueError("Challenge scene execution refused to regenerate character references; approved references must be reused")
        if any(r.get("approval") != "approved" for r in preflight["reused_images"] if r.get("kind") == "character"):
            raise ValueError("Challenge scene execution requires APPROVED character references")
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
    outcome = worker.execute(backend_factory, mode=worker_mode, smoke_scene="challenge", max_new_images=max_budget)
    sheet_path = contact_sheet(result_dir)
    validation = validate_one_click_result(worker.manifest_path, expected_request_hash=worker.request_hash)
    export_dir = work_root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    archive_base = export_dir / f"{result_dir.name}_{uuid.uuid4().hex[:12]}"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=result_dir))
    if outcome.get("status") != "complete":
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
    parser.add_argument("--mode", choices=("references", "challenge"), default="references")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples")
    parser.add_argument("--name")
    parser.add_argument("--characters", nargs="+", choices=("hare", "tortoise"), default=["hare", "tortoise"])
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "challenge":
        print(json.dumps(prepare_one_click_challenge_handoff(
            args.output_dir, name=args.name or DEFAULT_CHALLENGE_PACKAGE_NAME, attempt=args.attempt
        ), indent=2))
    else:
        print(json.dumps(prepare_one_click_handoff(
            args.output_dir, name=args.name or DEFAULT_PACKAGE_NAME, characters=args.characters, attempt=args.attempt
        ), indent=2))
