"""Portable request publishing and validated candidate import, using existing assets APIs."""
import base64
import json
import math
from pathlib import Path
import zipfile

from .asset_store import AssetStore, digest, read_image
from .assets import import_batch, review
from .characters import load_bible
from .image_batch import inside, validate_request, scene_seed
from .image_conditioning import CONDITIONING_REVISION, reference_layout
from .production_state import file_hash, now, write_json, production_directory
from .schema import load_story

MODULES = ("__init__.py", "asset_store.py", "characters.py", "schema.py", "production_state.py",
           "image_batch.py", "colab_images.py", "image_conditioning.py", "modern_images.py", "one_click.py")


def create_handoff(request_path, store, destination, *, extra_files=None):
    """Package the request + worker source + available references; never approve."""
    request_path, destination = Path(request_path), Path(destination)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    characters, _ = validate_request(request)
    catalog = store.read()
    entries, files, warnings = [], {}, []
    for identity, character in characters.items():
        encoded = character.get("approved_reference_png_base64")
        if encoded:
            payload = base64.b64decode(encoded, validate=True)
            checksum = digest(payload)
            if checksum != character["approved_reference_sha256"]:
                raise ValueError("Approved reference checksum mismatch while packaging")
            relative = f"references/{identity}_{checksum}.png"
            files[relative] = payload
            entries.append({"target_id": identity, "file": relative, "sha256": checksum,
                "asset_id": character["approved_asset_id"], "approval": "approved",
                "metadata": {**character.get("approved_reference_metadata", {}), "source": "current approved production request"}})
            continue
        for record in sorted(catalog["assets"].values(), key=lambda r: r["asset_id"]):
            if record["kind"] != "character" or record["target_id"] != identity:
                continue
            try:
                path = store.path_for(record)
                if file_hash(path) != record["sha256"]:
                    raise ValueError("candidate checksum mismatch")
                read_image(path)
            except (OSError, ValueError) as exc:
                warnings.append(f"Reference candidate {record['asset_id']} unavailable: {exc}")
                continue
            relative = f"references/{record['asset_id']}.png"
            files[relative] = path.read_bytes()
            entries.append({"target_id": identity, "file": relative, "sha256": record["sha256"],
                "asset_id": record["asset_id"], "approval": "candidate", "metadata": record["metadata"]})
    files["request.json"] = request_path.read_bytes()
    files["references.json"] = json.dumps({"version": 1, "references": entries}, indent=2).encode()
    for name in MODULES:
        files[f"src/{name}"] = Path(__file__).with_name(name).read_bytes()
    for name, payload in (extra_files or {}).items():
        inside(Path("."), name)  # Apply the existing safe-relative path rules.
        if name in files or name == "bundle_index.json" or not isinstance(payload, bytes):
            raise ValueError("Extra handoff files must be unique safe paths with byte payloads")
        files[name] = payload
    index = {"version": 1, "request_sha256": file_hash(request_path), "warnings": warnings,
             "files": {name: digest(payload) for name, payload in files.items()}}
    files["bundle_index.json"] = json.dumps(index, indent=2).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    return {"package_path": str(destination.resolve()), "package_sha256": file_hash(destination),
        "request_sha256": file_hash(request_path), "reference_count": len(entries), "warnings": warnings,
        "approval_required": True, "colab_started": False}


def validate_result(manifest_path, expected_request_hash=None):
    path = Path(manifest_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Result manifest must be an object")
    request_path = path.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    characters, scenes = validate_request(request)
    profile_hash = request.get("generation_config", {}).get("backend_profile_sha256")
    profile = None
    if profile_hash:
        profile_path = path.parent / "generation_profile.json"
        if file_hash(profile_path) != profile_hash:
            raise ValueError("Result generation profile checksum mismatch")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    actual_request_hash = file_hash(request_path)
    if data.get("version") != 1 or not isinstance(data.get("assets"), list) or data.get("request_sha256") != actual_request_hash:
        raise ValueError("Invalid result version/assets or request checksum")
    if expected_request_hash and actual_request_hash != expected_request_hash:
        raise ValueError("Result belongs to a different production request")
    if data.get("worker_version") in (1, 2) and (not isinstance(data.get("jobs"), dict) or
            any(not isinstance(job, dict) for job in data["jobs"].values())):
        raise ValueError("Worker result requires a jobs object of job records")
    seen, completed = set(), []
    for row in data["assets"]:
        if not isinstance(row, dict):
            raise ValueError("Result assets must be objects")
        kind, target = row["kind"], row["target_id"]
        if (kind, target) in seen or target not in (characters if kind == "character" else scenes if kind == "scene" else {}):
            raise ValueError("Duplicate or unknown result target")
        seen.add((kind, target))
        image_path = inside(path.parent, row["file"])
        if file_hash(image_path) != row["sha256"]:
            raise ValueError(f"Result image checksum mismatch: {kind}/{target}")
        image = read_image(image_path)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Result metadata must be an object")
        if profile and (metadata.get("model_id") != profile["model_id"] or
                        metadata.get("model_revision") != profile["model_revision"] or
                        metadata.get("model_license") != profile["model_license"]):
            raise ValueError("Result model provenance differs from its pinned generation profile")
        # Reused references retain original provenance; generated images require
        # full prompt/model/seed/timing metadata, including reference provenance.
        if not metadata.get("reused_reference"):
            for key in ("prompt", "seed", "model_id", "model_revision", "resolution", "runtime_seconds", "conditioning"):
                if key not in metadata:
                    raise ValueError(f"Missing generation metadata: {key}")
            if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int) or not 0 <= metadata["seed"] < 2**32:
                raise ValueError("Invalid image seed")
            runtime = metadata["runtime_seconds"]
            if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime < 0:
                raise ValueError("Invalid image runtime")
            if any(not isinstance(metadata[k], str) or not metadata[k] for k in ("model_id", "model_revision")):
                raise ValueError("Missing model identity/revision")
            if metadata["prompt"] != (characters if kind == "character" else scenes)[target]["prompt"] or list(image.size) != metadata["resolution"]:
                raise ValueError("Result prompt or resolution differs from its request/decoded image")
            expected_negative = (characters if kind == "character" else scenes)[target].get("negative_prompt", request["negative_prompt"])
            if metadata.get("negative_prompt", expected_negative) != expected_negative:
                raise ValueError("Result negative prompt differs from its request")
        if kind == "scene":
            identities = scenes[target]["characters"]
            expected = (CONDITIONING_REVISION if data.get("worker_version") == 2 else "ip_adapter_contact_sheet") if identities else "not_applicable"
            if metadata.get("conditioning") != expected:
                raise ValueError("Scene lacks required reference conditioning")
            if data.get("worker_version") == 2 and identities and metadata.get("conditioning_layout") != reference_layout(identities):
                raise ValueError("Scene reference placement does not match the declared conditioning layout")
            refs = metadata.get("references", {})
            if set(refs) != set(identities):
                raise ValueError("Scene reference metadata does not match character IDs")
            for reference in refs.values():
                ref_path = inside(path.parent, reference["file"])
                if file_hash(ref_path) != reference["sha256"]:
                    raise ValueError("Scene conditioning reference checksum mismatch")
                read_image(ref_path)
            if data.get("worker_version") in (1, 2) and metadata["seed"] != scene_seed(target, data["settings"]["base_seed"]):
                raise ValueError("Scene seed does not match the deterministic batch seed")
            completed.append(target)
    missing = [s for s in scenes if s not in completed]
    failed = [s for s in scenes if data.get("jobs", {}).get(f"scene:{s}", {}).get("status") == "failed"]
    if data.get("worker_version") in (1, 2):
        if not isinstance(data.get("jobs"), dict):
            raise ValueError("Worker result requires jobs object")
        expected_completion = {"completed_scene_ids": [s for s in scenes if s in completed], "missing_scene_ids": missing, "failed_scene_ids": failed}
        reference_only = request.get("generation_policy", {}).get("reference_only")
        if reference_only:
            completed_refs = [c for c in characters if ("character", c) in seen]
            missing_refs = [c for c in characters if c not in completed_refs]
            failed_refs = [c for c in characters if data["jobs"].get(f"character:{c}", {}).get("status") == "failed"]
            expected_completion.update(completed_reference_ids=completed_refs,
                missing_reference_ids=missing_refs, failed_reference_ids=failed_refs)
            if data.get("requested_reference_ids") != list(characters):
                raise ValueError("Requested reference IDs differ from the reference-only request")
            for row in data["assets"]:
                identity, metadata = row["target_id"], row["metadata"]
                if (data["jobs"].get(f"character:{identity}", {}).get("status") not in ("completed", "reused") or
                        metadata.get("reference_spec_revision") != characters[identity].get("reference_spec_revision") or
                        metadata.get("conditioning") != "text_only"):
                    raise ValueError("Reference-only result lacks a valid job/specification/text-only record")
                if not metadata.get("reused_reference") and metadata["seed"] != scene_seed(identity, data["settings"]["base_seed"]):
                    raise ValueError("Reference seed does not match the deterministic batch seed")
            if data.get("status") == "complete" and (missing_refs or failed_refs):
                raise ValueError("Batch claims complete while required references are missing/failed")
        if data.get("completion") != expected_completion:
            raise ValueError("Result completion summary does not match its actual images/jobs")
        for target in completed:
            if data.get("jobs", {}).get(f"scene:{target}", {}).get("status") != "completed":
                raise ValueError("Scene image has no completed job record")
    if data.get("status") == "complete" and (missing or failed):
        raise ValueError("Batch claims complete while requested scenes are missing/failed")
    result = {"manifest_path": str(path), "manifest_sha256": file_hash(path), "request_sha256": actual_request_hash,
        "status": data.get("status"), "asset_count": len(data["assets"]), "completed_scene_ids": completed,
        "missing_scene_ids": missing, "failed_scene_ids": failed, "errors": data.get("errors", []),
        "all_checksums_match": True, "all_images_decode": True}
    if request.get("generation_policy", {}).get("reference_only"):
        result.update({key: data["completion"][key] for key in
                       ("completed_reference_ids", "missing_reference_ids", "failed_reference_ids")})
    return result


def prepare_existing_production(production_dir, destination):
    """Read-only refresh of a handoff; never renders or changes production state."""
    root, destination = Path(production_dir).resolve(), Path(destination).resolve()
    production = json.loads((root / "production_manifest.json").read_text(encoding="utf-8"))
    if production.get("kind") != "story_studio_production":
        raise ValueError("Not an existing production project")
    request = Path(production["ai_generation_request"]["path"])
    if file_hash(request) != production["ai_generation_request"]["sha256"]:
        raise ValueError("Production request was modified; rerun production before exporting")
    destination.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory(prefix=".handoff_", dir=destination) as temp:
        result = create_handoff(request, AssetStore(production["input"]["asset_directory"]), Path(temp) / "package.zip")
        final = destination / f"request_{result['request_sha256'][:12]}_{result['package_sha256'][:12]}.zip"
        if final.exists():
            if file_hash(final) != result["package_sha256"]:
                raise ValueError("Existing published handoff differs; refusing to overwrite")
        else:
            Path(result["package_path"]).rename(final)
    return {**result, "package_path": str(final), "production_state_changed": False}


def receive_result(production_dir, manifest_path, *, asset_dir=None):
    """Validate everything before existing import/review APIs. No rendering/approval."""
    with production_directory(production_dir) as root:
        production = json.loads((root / "production_manifest.json").read_text(encoding="utf-8"))
        manifest_path = Path(manifest_path).resolve()
        archive_info = None
        if manifest_path.suffix.lower() == ".zip":
            archive_info = {"archive_path": str(manifest_path), "archive_sha256": file_hash(manifest_path)}
            extracted = root / "image_batch_inbox" / archive_info["archive_sha256"]
            with zipfile.ZipFile(manifest_path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or len(names) > 2000 or sum(m.file_size for m in archive.infolist()) > 256 * 1024 * 1024:
                    raise ValueError("Invalid or unexpectedly large batch archive")
                # Validate every path before writing anything. Never overwrite
                # extracted files; repeat intake must match identical bytes.
                for name in names:
                    inside(extracted, name)
                if "manifest.json" not in names or "request.json" not in names:
                    raise ValueError("Result ZIP must contain manifest.json and request.json at its root")
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    target = inside(extracted, member.filename)
                    payload = archive.read(member)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if target.read_bytes() != payload:
                            raise ValueError("Existing extracted result differs; refusing to overwrite")
                    else:
                        with target.open("xb") as stream:
                            stream.write(payload)
            manifest_path = extracted / "manifest.json"
        result = validate_result(manifest_path)
        if archive_info:
            result.update(archive_info)
        current_request = production["ai_generation_request"]["sha256"]
        known_requests = {current_request} | {stage["result"]["sha256"] for stage in production.get("cache", {}).values()
            if stage.get("name") == "ai_request" and stage.get("status") == "completed"}
        if result["request_sha256"] not in known_requests:
            raise ValueError("Result belongs to an unknown production request")
        result["request_is_current"] = result["request_sha256"] == current_request
        destination = root / "image_batch_results" / result["manifest_sha256"]
        receipt = destination / "receipt.json"
        if receipt.exists():
            return json.loads(receipt.read_text(encoding="utf-8")) | {
                "duplicate_result": True, "request_is_current": result["request_is_current"]}
        story = load_story(Path(production["input"]["script_path"]))
        bible = load_bible(Path(production["input"]["bible_path"]))
        store = AssetStore(asset_dir or production["input"]["asset_directory"])
        approvals_before = store.read()["approvals"]
        identities = import_batch(store, Path(manifest_path), story, bible)
        artifacts = review(store, destination / "review")
        if store.read()["approvals"] != approvals_before:
            raise RuntimeError("Import must not change approval pins")
        receipt_data = {**result, "imported_asset_ids": identities, "review_pages": artifacts["pages"],
            "review_json": str((destination / "review/review.json").resolve()), "imported_at": now(),
            "approved_automatically": False, "duplicate_result": False}
        write_json(receipt, receipt_data)
        (destination / "review_report.txt").write_text(
            "Batch imported as UNAPPROVED candidates. No image-quality approval was performed.\n"
            f"Status: {result['status']}\nCompleted scenes: {result['completed_scene_ids']}\n"
            f"Missing scenes: {result['missing_scene_ids']}\nFailed scenes: {result['failed_scene_ids']}\n"
            f"Errors: {result['errors']}\nReview full-resolution images and contact sheets before approving exact IDs.\n", encoding="utf-8")
        return receipt_data
