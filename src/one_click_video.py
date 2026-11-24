"""One-click I2V handoff packaging, verification, Colab execution, and ZIP export."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from .asset_store import AssetStore, digest
from .modern_video import WanI2VBackend, validate_i2v_profile
from .production_state import file_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_I2V_TEST_PACKAGE_NAME = "i2v_hare_sprints_v1_one_click"
I2V_WORKER_MODULES = (
    "__init__.py",
    "asset_store.py",
    "characters.py",
    "schema.py",
    "production_state.py",
    "modern_video.py",
    "one_click_video.py",
)


def _inside(root: Path, rel: str) -> Path:
    clean = PurePosixPath(rel)
    if clean.is_absolute() or ".." in clean.parts or not clean.parts:
        raise ValueError(f"Unsafe relative path: {rel}")
    target = (root / Path(*clean.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {rel}")
    return target


def prepare_one_click_i2v_handoff(
    destination=None,
    *,
    name=DEFAULT_I2V_TEST_PACKAGE_NAME,
    asset_dir=None,
    profile_path=None,
    overwrite=False,
):
    if not isinstance(name, str) or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple alphanumeric file name")
    destination = Path(destination or (ROOT / "examples")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    profile_file = Path(profile_path or (ROOT / "configs/one_click_i2v_test.json"))
    profile_bytes = profile_file.read_bytes()
    profile = validate_i2v_profile(json.loads(profile_bytes.decode("utf-8")))
    profile_sha256 = digest(profile_bytes)
    store = AssetStore(asset_dir or (ROOT / "assets/library"))

    scenes_payload = []
    keyframes = {}
    base_seed = profile["settings"]["base_seed"]
    for idx, (sid, spec) in enumerate(profile["scenes"].items()):
        approved = store.approved("scene", sid)
        if not approved:
            raise ValueError(f"Scene '{sid}' must be approved in AssetStore before preparing I2V handoff")
        record, keyframe_path = approved
        rel_keyframe = f"keyframes/{sid}{keyframe_path.suffix.lower()}"
        keyframes[rel_keyframe] = keyframe_path.read_bytes()
        scenes_payload.append({
            "target_id": sid,
            "keyframe_file": rel_keyframe,
            "keyframe_sha256": record["sha256"],
            "source_asset_id": record["asset_id"],
            "prompt": spec["prompt"].strip(),
            "negative_prompt": spec["negative_prompt"].strip(),
            "seed": base_seed + idx,
        })

    request = {
        "version": 1,
        "kind": "i2v_clips",
        "batch_id": f"{name}_{profile_sha256[:8]}",
        "model_id": profile["model_id"],
        "model_revision": profile["model_revision"],
        "model_license": profile["model_license"],
        "max_new_clips": profile["max_new_clips"],
        "scenes": scenes_payload,
    }
    request_bytes = json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request_sha256 = digest(request_bytes)
    req_txt = (ROOT / "notebooks/requirements-one-click.txt").read_bytes()

    bundle_files = {
        "request.json": request_bytes,
        "generation_profile.json": profile_bytes,
        "requirements-one-click.txt": req_txt,
        **keyframes,
    }
    src_root = ROOT / "src"
    for mod in I2V_WORKER_MODULES:
        bundle_files[f"src/{mod}"] = (src_root / mod).read_bytes()

    index = {
        "version": 1,
        "files": {rel: digest(data) for rel, data in sorted(bundle_files.items())},
    }
    index_bytes = json.dumps(index, indent=2, sort_keys=True).encode("utf-8")

    with tempfile.TemporaryDirectory(prefix="one_click_i2v_pkg_") as temp:
        temp_dir = Path(temp)
        req_out = temp_dir / f"{name}.json"
        req_out.write_bytes(request_bytes)
        zip_out = temp_dir / f"{name}.zip"
        with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel, data in sorted(bundle_files.items()):
                zf.writestr(rel, data)
            zf.writestr("bundle_index.json", index_bytes)
        copies = ((req_out, destination / f"{name}.json"), (zip_out, destination / f"{name}.zip"))
        if not overwrite:
            for src, dst in copies:
                if dst.exists() and file_hash(dst) != file_hash(src):
                    raise FileExistsError(f"Refusing to overwrite {dst}; pass overwrite=True")
        for src, dst in copies:
            shutil.copy2(src, dst)

    pkg_path = destination / f"{name}.zip"
    return {
        "package_path": str(pkg_path),
        "package_sha256": file_hash(pkg_path),
        "request_path": str(destination / f"{name}.json"),
        "request_sha256": request_sha256,
        "profile_sha256": profile_sha256,
        "scene_ids": [s["target_id"] for s in scenes_payload],
    }


def extract_and_verify_i2v_bundle(package_path, bundles_root, *, expected_package_sha256=None):
    package_path = Path(package_path)
    actual_pkg_sha = file_hash(package_path)
    if expected_package_sha256 and actual_pkg_sha != expected_package_sha256:
        raise ValueError(f"I2V package SHA256 mismatch: expected {expected_package_sha256}, got {actual_pkg_sha}")
    bundle_dir = Path(bundles_root).resolve() / actual_pkg_sha
    with zipfile.ZipFile(package_path) as zf:
        infos = zf.infolist()
        if len(infos) > 200 or sum(m.file_size for m in infos) > 128 * 1024 * 1024:
            raise ValueError("Unexpectedly large I2V handoff archive")
        index = json.loads(zf.read("bundle_index.json"))
        for rel, expected_sha in index["files"].items():
            data = zf.read(rel)
            if digest(data) != expected_sha:
                raise ValueError(f"I2V bundle file checksum mismatch: {rel}")
            target = _inside(bundle_dir, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    request = json.loads((bundle_dir / "request.json").read_text(encoding="utf-8"))
    profile = validate_i2v_profile(json.loads((bundle_dir / "generation_profile.json").read_text(encoding="utf-8")))
    for scene in request["scenes"]:
        kf = _inside(bundle_dir, scene["keyframe_file"])
        if digest(kf.read_bytes()) != scene["keyframe_sha256"]:
            raise ValueError(f"Keyframe checksum mismatch: {scene['target_id']}")
    return {"bundle_dir": bundle_dir, "request": request, "profile": profile}


def run_one_click_i2v_batch(bundle_dir, work_root, *, backend_factory, cache_dir):
    bundle_dir = Path(bundle_dir).resolve()
    work_root = Path(work_root).resolve()
    request = json.loads((bundle_dir / "request.json").read_text(encoding="utf-8"))
    profile = validate_i2v_profile(json.loads((bundle_dir / "generation_profile.json").read_text(encoding="utf-8")))
    if len(request["scenes"]) > profile["max_new_clips"]:
        raise ValueError("I2V request exceeds max_new_clips budget")

    batch_id = request["batch_id"]
    result_dir = work_root / "results" / batch_id
    clips_dir = result_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    backend = backend_factory(profile["settings"], None)
    assets = []
    for scene in request["scenes"]:
        sid = scene["target_id"]
        kf_path = _inside(bundle_dir, scene["keyframe_file"])
        out_mp4 = clips_dir / f"{sid}.mp4"
        meta = backend.generate_clip(
            kf_path,
            scene["prompt"],
            scene["negative_prompt"],
            out_mp4,
            seed=scene["seed"],
        )
        raw = out_mp4.read_bytes()
        if not raw or b"ftyp" not in raw[:64]:
            raise RuntimeError(f"Generated clip is not a valid MP4: {out_mp4}")
        assets.append({
            "kind": "clip",
            "target_id": sid,
            "file": f"clips/{out_mp4.name}",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "metadata": {**meta, "source_asset_id": scene["source_asset_id"], "keyframe_sha256": scene["keyframe_sha256"]},
        })

    manifest = {
        "version": 1,
        "batch_id": batch_id,
        "model_id": profile["model_id"],
        "model_revision": profile["model_revision"],
        "model_license": profile["model_license"],
        "assets": assets,
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    outcome = {"status": "complete", "batch_id": batch_id, "assets": assets, "errors": []}
    (result_dir / "outcome.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8", newline="\n")

    archive_path = work_root / "results" / f"{batch_id}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(result_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(result_dir).as_posix())
    return {
        "result_dir": result_dir,
        "manifest_path": manifest_path,
        "archive_path": archive_path,
        "archive_sha256": file_hash(archive_path),
        "outcome": outcome,
    }
