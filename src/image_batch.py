"""Shared, CPU-testable Colab batch state machine. No model imports or API calls.

Generated files are candidates. Batch preview consent is never asset approval.
"""
import base64
from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
import time
import uuid

from PIL import Image

from .asset_store import digest, read_image
from .characters import safe_relative, PROMPT_REVISION
from .production_state import file_hash, now, write_json, fingerprint
from .schema import _identifier
from .image_conditioning import CONDITIONING_REVISION, reference_layout

WORKER_VERSION = 2


def inside(root, relative):
    path = (Path(root) / safe_relative(relative)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Batch file escapes its folder")
    return path


def scene_seed(identity, base_seed=20261123):
    import hashlib
    _identifier(identity)
    return (base_seed + int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)) % 2**32


def validate_request(request):
    if not isinstance(request, dict) or request.get("version") != 1:
        raise ValueError("Expected an existing version 1 Colab request")
    for key in ("title", "negative_prompt", "style", "environment"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError(f"Request requires {key} text")
    characters, scenes = {}, {}
    for key, target in (("characters", characters), ("scenes", scenes)):
        if not isinstance(request.get(key), list):
            raise ValueError(f"Request requires {key} list")
        for row in request[key]:
            if not isinstance(row, dict):
                raise ValueError("Request rows must be objects")
            identity = _identifier(row.get("id"))
            if identity in target or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"Duplicate ID or missing prompt: {key}/{identity}")
            target[identity] = row
    for scene in scenes.values():
        identities = scene.get("characters")
        if not isinstance(identities, list) or any(not isinstance(c, str) or c not in characters for c in identities) or len(set(identities)) != len(identities):
            raise ValueError("Scene character IDs must exist uniquely in the request")
    locked = request.get("locked_scene_ids", [])
    if not isinstance(locked, list) or any(not isinstance(s, str) for s in locked):
        raise ValueError("Invalid locked scene ids")
    for field in ("generation_policy", "generation_config"):
        if field in request and not isinstance(request[field], dict):
            raise ValueError(f"{field} must be an object")
    allowed = request.get("generation_policy", {}).get("allowed_scene_ids")
    if allowed is not None and (not isinstance(allowed, list) or set(scenes) - set(allowed)):
        raise ValueError("Request scenes exceed the explicit smoke-only scope")
    policy = request.get("generation_policy", {})
    for key in ("reference_only", "fresh_references"):
        if key in policy and not isinstance(policy[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if policy.get("reference_only"):
        if scenes or not characters or policy.get("allowed_scene_ids") != [] or policy.get("full_batch_enabled") is not False:
            raise ValueError("Reference-only requests require characters, zero scenes and disabled scene generation")
        if policy.get("required_reference_ids") != list(characters):
            raise ValueError("Reference-only required_reference_ids must match its character rows")
    return characters, scenes


@dataclass(frozen=True)
class BatchSettings:
    family: str = "sdxl"
    base_seed: int = 20261123
    steps: int = 24
    adapter_scale: float = .35
    conditioning: str = CONDITIONING_REVISION
    prompt_revision: str = PROMPT_REVISION
    reference_size: int = 768  # Preserve existing scene workflows; v3 requests explicitly use 1024.

    def validate(self):
        if self.conditioning != CONDITIONING_REVISION or self.prompt_revision != PROMPT_REVISION:
            raise ValueError("Use the current explicit prompting/conditioning revision in a NEW result batch")
        if self.family not in ("sdxl", "sd15", "flux2_klein"):
            raise ValueError("Choose sdxl, sd15 or flux2_klein explicitly")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int) or not 0 <= self.base_seed < 2**32:
            raise ValueError("Invalid base seed")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 100:
            raise ValueError("Invalid inference steps")
        if isinstance(self.adapter_scale, bool) or not isinstance(self.adapter_scale, (int, float)) or not 0 < self.adapter_scale <= 1:
            raise ValueError("Reference conditioning must remain enabled with 0 < adapter_scale <= 1")
        if isinstance(self.reference_size, bool) or not isinstance(self.reference_size, int) or self.reference_size not in (512, 768, 1024):
            raise ValueError("reference_size must be 512, 768 or 1024")

    def size(self, kind):
        return ((512, 512) if kind == "character" else (768, 432)) if self.family == "sd15" else ((self.reference_size, self.reference_size) if kind == "character" else (1024, 576))


def generation_identity(request_hash, settings):
    """Request, settings and generation implementation all bind cache identity."""
    implementation = {name: file_hash(Path(__file__).with_name(name)) for name in
                      ("image_batch.py", "colab_images.py", "image_conditioning.py", "characters.py")}
    if settings.family == "flux2_klein":
        implementation.update({name: file_hash(Path(__file__).with_name(name)) for name in ("modern_images.py", "one_click.py")})
    return fingerprint({"request_sha256": request_hash, "settings": asdict(settings), "implementation": implementation})


def reference_sources(manifest_path):
    """Read either the portable reference index or a legacy/persistent result batch.

    A result's character references are candidates unless their checksum matches
    an approved reference in the CURRENT production request.
    """
    path = Path(manifest_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("references") if "references" in data else [a for a in data.get("assets", []) if a["kind"] == "character"]
    if not isinstance(rows, list):
        raise ValueError("Reference source requires a references or assets list")
    result = []
    for row in rows:
        identity = _identifier(row.get("target_id"))
        source = inside(path.parent, row["file"])
        if file_hash(source) != row["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {identity}")
        read_image(source)
        result.append({**row, "target_id": identity, "source_path": str(source),
                       "source_manifest": str(path), "approval": "candidate"})
    return result


class BatchWorker:
    def __init__(self, request_path, result_dir, *, references=None, settings=None):
        self.request_path, self.root = Path(request_path).resolve(), Path(result_dir).resolve()
        self.request_bytes = self.request_path.read_bytes()
        self.request = json.loads(self.request_bytes)
        self.characters, self.scenes = validate_request(self.request)
        self.request_hash = digest(self.request_bytes)
        self.settings = settings or BatchSettings()
        self.settings.validate()
        requested_config = self.request.get("generation_config", {})
        if any(requested_config.get(key, getattr(self.settings, key)) != getattr(self.settings, key)
               for key in ("conditioning", "prompt_revision", "reference_size", "base_seed", "family", "steps")):
            raise ValueError("Request prompting/conditioning configuration is incompatible with this worker; prepare a NEW request/batch")
        self.generation_identity = generation_identity(self.request_hash, self.settings)
        self.references = list(references or [])
        self.manifest_path = self.root / "manifest.json"
        self.data = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {
            "version": 1, "worker_version": WORKER_VERSION, "batch_id": self.root.name, "status": "prepared", "assets": [],
            "jobs": {}, "errors": [], "runs": [], "created_at": now(), "request_sha256": self.request_hash,
            "settings": asdict(self.settings), "generation_identity": self.generation_identity,
            "requested_scene_ids": list(self.scenes)}
        if self.request.get("generation_policy", {}).get("reference_only"):
            self.data["requested_reference_ids"] = list(self.characters)
        if self.data.get("worker_version") != WORKER_VERSION or self.data.get("generation_identity") != self.generation_identity:
            raise ValueError("Resume request/settings differ or generation implementation changed. Use a NEW result directory; existing images will not be overwritten.")
        for identity in self.scenes:
            self.data["jobs"].setdefault(f"scene:{identity}", {"status": "pending", "seed": scene_seed(identity, self.settings.base_seed)})
        if self.request.get("generation_policy", {}).get("reference_only"):
            for identity in self.characters:
                self.data["jobs"].setdefault(f"character:{identity}", {"status": "pending", "seed": scene_seed(identity, self.settings.base_seed)})

    def refresh(self):
        if self.manifest_path.exists():
            latest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if latest.get("worker_version") != WORKER_VERSION or latest.get("generation_identity") != self.generation_identity:
                raise ValueError("Persistent batch request/settings changed; choose a NEW result directory")
            self.data = latest
            for identity in self.scenes:
                self.data["jobs"].setdefault(f"scene:{identity}", {"status": "pending", "seed": scene_seed(identity, self.settings.base_seed)})

    def save(self):
        self.data["updated_at"] = now()
        write_json(self.manifest_path, self.data)
        write_json(self.root / "history" / f"{time.time_ns()}_{uuid.uuid4().hex[:8]}.json", self.data)

    def _asset(self, kind, identity):
        for record in self.data["assets"]:
            if record["kind"] == kind and record["target_id"] == identity:
                try:
                    path = inside(self.root, record["file"])
                    if file_hash(path) == record["sha256"]:
                        read_image(path)
                        return record
                except (OSError, ValueError):
                    pass
        return None

    def _sources(self):
        sources = list(self.references)
        for identity, character in self.characters.items():
            encoded = character.get("approved_reference_png_base64")
            if encoded:
                payload = base64.b64decode(encoded, validate=True)
                if digest(payload) != character["approved_reference_sha256"]:
                    raise ValueError("Approved embedded reference checksum mismatch")
                with Image.open(BytesIO(payload)) as image:
                    image.load()
                    if min(image.size) < 64 or image.width * image.height > 16_000_000:
                        raise ValueError("Invalid embedded reference dimensions")
                sources.append({"target_id": identity, "sha256": digest(payload), "payload": payload,
                    "asset_id": character["approved_asset_id"], "metadata": character.get("approved_reference_metadata", {"origin": "current approved request"})})
            previous = self._asset("character", identity)
            if previous:
                sources.append({**previous, "source_path": str(inside(self.root, previous["file"]))})
        for source in sources:
            character = self.characters.get(source["target_id"], {})
            source["approval"] = "approved" if character.get("approved_reference_sha256") == source["sha256"] else "candidate"
        return sources

    def preflight(self, *, mode="dry_run", smoke_scene=None, max_new_images=3,
                  allow_unapproved_references=False, quality_reviewed=False, reference_ids=None, references_reviewed=False):
        self.refresh()
        if mode not in ("dry_run", "references", "smoke", "full"):
            raise ValueError("Mode must be dry_run, references, smoke, or full")
        if any(not isinstance(value, bool) for value in (allow_unapproved_references, quality_reviewed, references_reviewed)):
            raise ValueError("Preview consent and quality review must be explicit booleans")
        if isinstance(max_new_images, bool) or not isinstance(max_new_images, int) or max_new_images < 0:
            raise ValueError("max_new_images must be a nonnegative integer")
        smoke_scene = smoke_scene or next(iter(self.scenes), None)
        if self.scenes and smoke_scene not in self.scenes:
            raise ValueError("Smoke scene is absent from the request")
        selected = [] if mode == "references" else [smoke_scene] if mode == "smoke" and smoke_scene else list(self.scenes)
        blockers, warnings, new, reused, chosen = [], [], [], [], {}
        policy = self.request.get("generation_policy", {})
        if policy.get("reference_only") and mode not in ("dry_run", "references"):
            blockers.append("Reference-only request: scene generation is disabled; use MODE='references' or 'dry_run'")
        if mode == "full" and policy.get("full_batch_enabled") is False:
            blockers.append("This request is restricted to the challenge smoke. Full-batch generation is disabled pending human quality review.")
        if mode == "smoke" and policy.get("require_reference_review") and not references_reviewed:
            blockers.append("Inspect the NEW single-view references first, then explicitly set REFERENCES_REVIEWED=True")
        locked = set(selected) & set(self.request.get("locked_scene_ids", []))
        if locked:
            blockers.append(f"Scene approvals remain locked: {sorted(locked)}; unlock locally and export a fresh request")
        for identity in selected:
            reference_layout(self.scenes[identity]["characters"])  # Validate strategy limits before model loading.
        sources = self._sources()
        if policy.get("fresh_references"):
            # Fresh requests never inherit rejected external candidates. Only
            # this exact request/settings/code batch's successful work resumes.
            own = {c: self._asset("character", c) for c in self.characters}
            sources = [r for r in sources if own.get(r["target_id"]) and
                       r["sha256"] == own[r["target_id"]]["sha256"]]
            if self.references:
                warnings.append("Fresh-reference request: external references are preserved but not reused")
        required_spec = policy.get("reference_spec_revision")
        if required_spec:
            historical = [r for r in sources if r.get("metadata", {}).get("reference_spec_revision") != required_spec]
            if historical:
                warnings.append("Historical reference candidates are preserved but excluded: they do not meet this request's single-view specification revision")
            sources = [r for r in sources if r not in historical]
        needed = sorted(self.characters) if policy.get("reference_only") else sorted({c for s in self.scenes.values() for c in s["characters"]})
        reference_ids = reference_ids or {}
        for identity in needed:
            options = [r for r in sources if r["target_id"] == identity]
            for row in options:
                if "source_path" in row:
                    if file_hash(row["source_path"]) != row["sha256"]:
                        raise ValueError(f"Reference changed: {identity}")
                    read_image(Path(row["source_path"]))
            preferred = [r for r in options if r["approval"] == "approved"]
            choice = reference_ids.get(identity)
            if choice:
                preferred = [r for r in options if r.get("asset_id", r["sha256"]) == choice or r["sha256"] == choice]
                if not preferred:
                    blockers.append(f"Selected reference is absent: {identity}/{choice}")
            unique = {r["sha256"]: r for r in (preferred or options)}
            if len(unique) == 1:
                row = next(iter(unique.values()))
                chosen[identity] = row
                reused.append({"kind": "character", "id": identity, "sha256": row["sha256"], "approval": row["approval"]})
                if row["approval"] != "approved":
                    warnings.append(f"EXPERIMENTAL PREVIEW: {identity} is UNAPPROVED; reuse does not approve it")
                    if not allow_unapproved_references and mode != "references" and not policy.get("reference_only"):
                        blockers.append(f"Unapproved reference {identity} requires explicit preview opt-in; it will not be regenerated automatically")
            elif unique:
                blockers.append(f"Multiple references for {identity}; choose its exact reference ID/checksum")
            elif not choice:
                new.append({"kind": "character", "id": identity})
                if not allow_unapproved_references and mode != "references" and not policy.get("reference_only"):
                    blockers.append(f"New reference {identity} would be unapproved; enable experimental preview or supply approved references")
        for record in self.data["assets"]:
            if record["kind"] == "scene":
                for identity, reference in record.get("metadata", {}).get("references", {}).items():
                    if identity in chosen and chosen[identity]["sha256"] != reference["sha256"]:
                        blockers.append("Reference binding changed after scene generation. Use a NEW result batch, never reuse incompatible scenes.")
        for identity in selected:
            if self._asset("scene", identity):
                reused.append({"kind": "scene", "id": identity})
            else:
                new.append({"kind": "scene", "id": identity})
        if mode == "smoke" and policy.get("require_reference_review") and any(task["kind"] == "character" for task in new):
            blockers.append("Generate and review references separately in MODE='references' before the challenge smoke")
        if len(new) > max_new_images:
            blockers.append(f"Planned {len(new)} new images exceeds limit {max_new_images}")
        if mode == "full" and new:
            if not policy.get("reviewed_smoke_sha256"):
                smoke = self._asset("scene", smoke_scene) if smoke_scene else None
                if not smoke or not self.data.get("smoke_scene_id") or not quality_reviewed:
                    blockers.append("Run one successful smoke scene first, inspect it, then explicitly set QUALITY_REVIEWED=True")
        return {"request_sha256": self.request_hash, "mode": mode, "requested_scene_ids": list(self.scenes),
            "required_reference_ids": needed,
            "generation_identity": self.generation_identity,
            "selected_scene_ids": selected, "smoke_scene_id": smoke_scene, "new_images": new,
            "new_image_count": len(new), "reused_images": reused, "warnings": warnings,
            "blockers": list(dict.fromkeys(blockers)), "chosen_references": chosen}

    def _completion(self):
        completed = [s for s in self.scenes if self._asset("scene", s)]
        failed = [s for s in self.scenes if self.data["jobs"].get(f"scene:{s}", {}).get("status") == "failed"]
        missing = [s for s in self.scenes if s not in completed]
        self.data["completion"] = {"completed_scene_ids": completed, "missing_scene_ids": missing, "failed_scene_ids": failed}
        if self.request.get("generation_policy", {}).get("reference_only"):
            references = [c for c in self.characters if self._asset("character", c)]
            missing_refs = [c for c in self.characters if c not in references]
            failed_refs = [c for c in self.characters if self.data["jobs"].get(f"character:{c}", {}).get("status") == "failed"]
            self.data["completion"].update(completed_reference_ids=references,
                missing_reference_ids=missing_refs, failed_reference_ids=failed_refs)
            return not missing_refs and not failed_refs
        return not missing and not failed

    def _register_reference(self, identity, source):
        previous = self._asset("character", identity)
        if previous and previous["sha256"] == source["sha256"]:
            return
        payload = source.get("payload") or Path(source["source_path"]).read_bytes()
        path = self.root / "inputs" / f"{identity}_{source['sha256'][:16]}_{uuid.uuid4().hex[:8]}.png"
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
        metadata = dict(source.get("metadata", {}))
        metadata["reused_reference"] = {"approval": source["approval"], "source_manifest": source.get("source_manifest"),
            "asset_id": source.get("asset_id"), "sha256": source["sha256"]}
        record = {"kind": "character", "target_id": identity, "file": path.relative_to(self.root).as_posix(),
                  "sha256": file_hash(path), "metadata": metadata}
        self.data["assets"] = [r for r in self.data["assets"] if (r["kind"], r["target_id"]) != ("character", identity)] + [record]
        self.data["jobs"][f"character:{identity}"] = {"status": "reused", "sha256": record["sha256"]}
        self.save()

    def execute(self, backend_factory, **controls):
        plan = self.preflight(**controls)
        if plan["mode"] == "dry_run":
            return {k: v for k, v in plan.items() if k != "chosen_references"}  # No writes, no model loading.
        if plan["blockers"]:
            raise ValueError("Preflight blocked: " + "; ".join(plan["blockers"]))
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".worker.lock"
        try:
            handle = lock.open("x")
        except FileExistsError as exc:
            raise RuntimeError("Batch worker is already running; inspect any stale lock before resuming") from exc
        try:
            handle.close()
            plan = self.preflight(**controls)  # Recheck after locking; another session may have finished work.
            if plan["blockers"]:
                raise ValueError("Preflight blocked: " + "; ".join(plan["blockers"]))
            snapshot = self.root / "request.json"
            if snapshot.exists() and file_hash(snapshot) != self.request_hash:
                raise ValueError("Persistent request snapshot was modified")
            if not snapshot.exists():
                with snapshot.open("xb") as stream:
                    stream.write(self.request_bytes)
            self.data["status"] = "running"
            for previous in self.data["runs"]:
                if previous.get("status") == "running":
                    previous["status"] = "interrupted"
            run = {"mode": plan["mode"], "status": "running", "started_at": now(), "new_image_count": plan["new_image_count"],
                   "warnings": plan["warnings"], "generated": [], "reused": plan["reused_images"],
                   "references_reviewed": controls.get("references_reviewed", False),
                   "experimental_reference_consent": controls.get("allow_unapproved_references", False)}
            self.data["runs"].append(run)
            self.save()
            for identity, source in plan["chosen_references"].items():
                self._register_reference(identity, source)
            backend = None
            for task in plan["new_images"]:
                kind, identity = task["kind"], task["id"]
                key = f"{kind}:{identity}"
                job = {"kind": kind, "id": identity, "seed": scene_seed(identity, self.settings.base_seed),
                    "prompt": (self.characters if kind == "character" else self.scenes)[identity]["prompt"],
                    "negative_prompt": (self.characters if kind == "character" else self.scenes)[identity].get("negative_prompt", self.request["negative_prompt"]),
                    "size": self.settings.size(kind),
                    "conditioning": self.settings.conditioning if kind == "scene" and self.scenes[identity]["characters"] else "not_applicable" if kind == "scene" else "text_only"}
                state = {"status": "running", "started_at": now(), "seed": job["seed"],
                         "attempt": self.data["jobs"].get(key, {}).get("attempt", 0) + 1}
                self.data["jobs"][key] = state
                self.save()
                start = time.perf_counter()
                try:
                    if backend is None:
                        backend = backend_factory(self.settings, self.data.get("backend"))
                        self.data["backend"] = backend.provenance
                        self.save()
                    refs, ref_metadata = {}, {}
                    if kind == "scene":
                        for character in self.scenes[identity]["characters"]:
                            asset = self._asset("character", character)
                            if not asset:
                                raise ValueError(f"Reference missing before scene generation: {character}")
                            refs[character] = read_image(inside(self.root, asset["file"]))
                            ref_metadata[character] = {"file": asset["file"], "sha256": asset["sha256"],
                                "approval": plan["chosen_references"].get(character, {}).get("approval", "candidate")}
                    image, metadata = backend.generate(job, refs)
                    expected_conditioning = job["conditioning"]
                    if metadata.get("conditioning") != expected_conditioning:
                        raise ValueError("Backend did not apply the required reference conditioning")
                    if refs and metadata.get("conditioning_layout") != reference_layout(refs):
                        raise ValueError("Backend reference conditioning layout differs from the requested spatial gates")
                    if image.size != tuple(job["size"]):
                        raise ValueError("Backend returned incorrect image resolution")
                    path = self.root / "images" / f"{kind}_{identity}_{job['seed']}_{uuid.uuid4().hex[:8]}.png"
                    path.parent.mkdir(exist_ok=True)
                    with path.open("xb") as stream:
                        image.convert("RGB").save(stream, format="PNG")
                    record = {"kind": kind, "target_id": identity, "file": path.relative_to(self.root).as_posix(),
                        "sha256": file_hash(path), "metadata": {**metadata, "prompt": job["prompt"],
                            "negative_prompt": job["negative_prompt"], "seed": job["seed"], "resolution": list(image.size),
                            "runtime_seconds": round(time.perf_counter() - start, 4), "settings": asdict(self.settings),
                            "references": ref_metadata, "experimental_preview": any(r["approval"] != "approved" for r in ref_metadata.values())}}
                    if kind == "character":
                        record["metadata"]["reference_spec_revision"] = self.characters[identity].get("reference_spec_revision")
                    elif refs:
                        record["metadata"]["conditioning_layout"] = reference_layout(refs)
                    self.data["assets"] = [r for r in self.data["assets"] if (r["kind"], r["target_id"]) != (kind, identity)] + [record]
                    state.update(status="completed", file=record["file"], sha256=record["sha256"], completed_at=now())
                    run["generated"].append(key)
                    if plan["mode"] == "smoke" and kind == "scene":
                        self.data["smoke_scene_id"] = identity
                    self._completion()
                    self.save()  # Every successful image is persistent before the next job.
                except BaseException as exc:
                    state.update(status="failed", error=str(exc), error_type=type(exc).__name__, completed_at=now())
                    self.data["errors"].append({"job": key, "at": now(), "error": str(exc), "type": type(exc).__name__})
                    self.data["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    run.update(status=self.data["status"], completed_at=now())
                    self._completion()
                    self.save()
                    if not isinstance(exc, Exception):
                        raise
                    break  # No automatic costly retry or silent unconditioned fallback.
            complete = self._completion()
            if complete:
                self.data["status"] = "complete"
            elif self.data["status"] == "running":
                self.data["status"] = "references_ready" if plan["mode"] == "references" else "awaiting_smoke_review" if plan["mode"] == "smoke" else "partial"
            if controls.get("quality_reviewed") and self.scenes:
                smoke = self._asset("scene", self.data.get("smoke_scene_id"))
                self.data["smoke_quality_review"] = {"at": now(), "explicit_user_confirmation": True,
                    "sha256": smoke["sha256"] if smoke else None, "asset_approval": False}
            run.update(completed_at=now(), status=self.data["status"])
            self.save()
            return self.data
        finally:
            lock.unlink(missing_ok=True)


def contact_sheet(result_dir):
    """Review-only montage, including explicit missing/failed scene panels."""
    from PIL import ImageDraw, ImageFont, ImageOps
    root = Path(result_dir)
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    targets = [("character", c["id"]) for c in request["characters"]] + [("scene", s["id"]) for s in request["scenes"]]
    sheet = Image.new("RGB", (1200, 70 + 360 * max(1, (len(targets) + 2) // 3)), "#f5f6f8")
    draw = ImageDraw.Draw(sheet)
    face = ImageFont.load_default(size=20)
    draw.text((20, 20), f"UNAPPROVED BATCH REVIEW | {data['status']} | {root.name}", font=face, fill="#263f3c")
    for n, (kind, identity) in enumerate(targets):
        x, y = n % 3 * 400, 70 + n // 3 * 360
        record = next((r for r in data["assets"] if r["kind"] == kind and r["target_id"] == identity), None)
        status = data["jobs"].get(f"{kind}:{identity}", {}).get("status", "MISSING")
        draw.text((x + 12, y + 8), f"{kind}/{identity} | {status}", font=face, fill="#263f3c")
        if record:
            try:
                thumb = ImageOps.contain(read_image(inside(root, record["file"])), (376, 300))
                sheet.paste(thumb, (x + (400 - thumb.width) // 2, y + 42))
            except (OSError, ValueError):
                draw.text((x + 20, y + 140), "INVALID / NEEDS REPAIR", fill="#a23b22", font=face)
        else:
            draw.text((x + 40, y + 140), "NO GENERATED IMAGE", fill="#a23b22", font=face)
    destination = root / "reviews" / f"contact_sheet_{uuid.uuid4().hex[:12]}.jpg"
    destination.parent.mkdir(exist_ok=True)
    sheet.save(destination, quality=92)
    return destination
