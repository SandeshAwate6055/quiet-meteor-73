"""Atomic production checkpoints and content-checked stage reuse.

Stage attempts use new directories. Successful media is never overwritten.
Only this production's own manifest is updated; run snapshots are retained.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


@contextmanager
def production_directory(root):
    root = Path(root).resolve()
    manifest = root / "production_manifest.json"
    if root.exists() and any(root.iterdir()):
        if not manifest.exists():
            raise ValueError("Output directory contains existing files and is not an orchestrator project. Choose a NEW output directory.")
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("kind") != "story_studio_production" or existing.get("version") != 1:
            raise ValueError("Output manifest belongs to a different or unsupported project")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".production.lock"
    try:
        handle = lock.open("x")
    except FileExistsError as exc:
        raise RuntimeError(f"Production is locked: {lock}. After a crash, verify no writer is active before removing this lock.") from exc
    try:
        handle.write(json.dumps({"started_at": now()}))
        handle.close()
        yield root
    finally:
        lock.unlink(missing_ok=True)


class ProductionState:
    def __init__(self, root):
        self.root = root
        self.path = root / "production_manifest.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "kind": "story_studio_production", "version": 1, "project_id": uuid.uuid4().hex,
            "created_at": now(), "cache": {}, "runs": []}
        for previous in self.data["runs"]:
            if previous["status"] == "running":
                previous["status"] = "interrupted"
        for key in ("title", "input", "settings", "plan", "assets", "asset_summary", "providers",
                    "ai_generation_request", "image_batch_handoff", "timing", "quality"):
            self.data.pop(key, None)  # A new failed run must not inherit old success claims.
        self.run = {"id": uuid.uuid4().hex, "started_at": now(), "status": "running", "executed_stages": [], "reused_stages": []}
        self.data["runs"].append(self.run)
        self.data.update(status="running", scenes=[], outputs={}, warnings=[], failure=None, active_stage="planning")
        self.save()

    def save(self):
        self.data["updated_at"] = now()
        write_json(self.path, self.data)

    def _valid(self, record):
        try:
            return record["status"] == "completed" and isinstance(record.get("result"), dict) and bool(record["artifacts"]) and all(
                Path(a["path"]).resolve().is_relative_to(self.root) and file_hash(a["path"]) == a["sha256"]
                for a in record["artifacts"])
        except (OSError, KeyError):
            return False

    def stage(self, name, dependencies, action, validate=None):
        key = fingerprint({"stage": name, "dependencies": dependencies})
        cache = self.data["cache"]
        self.data["active_stage"] = name
        if key in cache and self._valid(cache[key]):
            record = cache[key]
            try:
                if validate:
                    validate(record["result"])
            except (OSError, ValueError, RuntimeError):
                pass
            else:
                self.run["reused_stages"].append(name)
                self.save()
                return record["result"], "reused"
        directory = self.root / "work" / key[:16] / uuid.uuid4().hex[:12]
        directory.mkdir(parents=True, exist_ok=False)
        record = {"name": name, "status": "running", "dependencies": dependencies,
                  "started_at": now(), "directory": str(directory), "attempt": cache.get(key, {}).get("attempt", 0) + 1}
        cache[key] = record
        self.save()
        start = time.perf_counter()
        try:
            result, paths = action(directory)
            if validate:
                validate(result)
            artifacts = []
            for path in paths:
                path = Path(path).resolve()
                if not path.is_relative_to(directory) or not path.is_file():
                    raise ValueError("Provider must return new artifacts within its stage directory")
                artifacts.append({"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size})
            record.update(status="completed", result=result, artifacts=artifacts, completed_at=now(),
                          runtime_seconds=round(time.perf_counter() - start, 4))
            self.run["executed_stages"].append(name)
            self.save()
            return result, "generated"
        except BaseException as exc:
            record.update(status="failed", error=str(exc), error_type=type(exc).__name__,
                          completed_at=now(), runtime_seconds=round(time.perf_counter() - start, 4))
            self.save()
            raise

    def finish(self, error=None):
        status = "completed" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        self.data["status"] = self.run["status"] = status
        self.run["completed_at"] = now()
        self.run["elapsed_seconds"] = (datetime.fromisoformat(self.run["completed_at"]) - datetime.fromisoformat(self.run["started_at"])).total_seconds()
        if error is not None:
            self.data["failure"] = {"stage": self.data["active_stage"], "message": str(error), "type": type(error).__name__}
            self.run["failure"] = self.data["failure"]
        self.save()
        write_json(self.root / "runs" / f"{self.run['id']}.json", self.data)
