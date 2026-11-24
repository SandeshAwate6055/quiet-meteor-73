"""Append-only candidates and explicitly pinned approvals; no ML dependencies.

Locks are application-level, not filesystem access controls. Checksums detect manual edits.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import uuid

from PIL import Image, ImageOps

from .characters import safe_relative
from .schema import _identifier


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if min(image.size) < 64 or max(image.size) > 8192 or image.width * image.height > 16_000_000:
            raise ValueError("Images must be at least 64x64, at most 8192 per side and 16 megapixels")
        return ImageOps.exif_transpose(image).convert("RGB")


VALID_ASSET_KINDS = ("scene", "character", "clip")


class AssetStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def read(self):
        path = self.root / "catalog.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "version": 1, "assets": {}, "approvals": {"scene": {}, "character": {}}, "history": []}
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported asset catalog")
        if not isinstance(data.get("assets"), dict) or not isinstance(data.get("approvals"), dict) or not isinstance(data.get("history"), list):
            raise ValueError("Malformed asset catalog")
        if any(not isinstance(data["approvals"].get(kind), dict) for kind in ("scene", "character")):
            raise ValueError("Malformed approval registry")
        if "clip" in data["approvals"] and not isinstance(data["approvals"]["clip"], dict):
            raise ValueError("Malformed clip approval registry")
        return data

    @contextmanager
    def edit(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".write.lock"
        try:
            handle = lock.open("x")
        except FileExistsError as exc:
            raise RuntimeError(f"Asset store is busy: {lock}. If a writer crashed, inspect before removing its stale lock.") from exc
        try:
            handle.close()
            data = self.read()
            yield data
            temp = self.root / f".catalog_{uuid.uuid4().hex}.tmp"
            temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp.replace(self.root / "catalog.json")
        finally:
            lock.unlink(missing_ok=True)

    def path_for(self, record):
        path = (self.root / safe_relative(record["file"])).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Asset path escapes store")
        return path

    def import_image(self, source: Path, kind: str, target_id: str, metadata=None):
        if kind not in ("scene", "character"):
            raise ValueError("Asset kind must be scene or character")
        _identifier(target_id)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Asset metadata must be an object")
        image = read_image(source)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = buffer.getvalue()
        checksum = digest(encoded)
        asset_id = f"{kind}_{target_id}_{checksum}"
        record = {"asset_id": asset_id, "kind": kind, "target_id": target_id,
                  "file": f"candidates/{asset_id}.png", "sha256": checksum,
                  "source_sha256": digest(source.read_bytes()), "resolution": list(image.size),
                  "metadata": metadata or {}, "imported_at": datetime.now(timezone.utc).isoformat()}
        with self.edit() as catalog:
            if asset_id in catalog["assets"]:
                existing = catalog["assets"][asset_id]
                if digest(self.path_for(existing).read_bytes()) != checksum:
                    raise ValueError("Existing candidate was modified; refusing to replace it")
                return asset_id
            path = self.path_for(record)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive create: never overwrite a previous image, even an orphan.
            with path.open("xb") as stream:
                stream.write(encoded)
            catalog["assets"][asset_id] = record
        return asset_id

    def import_clip(self, source: Path, target_id: str, metadata=None):
        """Import an MP4 animation clip candidate with checksum and provenance."""
        target_id = _identifier(target_id)
        source = Path(source)
        if not source.is_file() or source.suffix.lower() != ".mp4":
            raise ValueError("Clip candidates must be existing .mp4 files")
        raw = source.read_bytes()
        if not raw or len(raw) > 512 * 1024 * 1024 or b"ftyp" not in raw[:64]:
            raise ValueError("Clip candidate must be a valid non-empty MP4 file")
        sha = digest(raw)
        asset_id = f"clip:{target_id}:{sha[:16]}"
        destination = (self.root / "candidates" / f"clip_{target_id}_{sha[:16]}.mp4").resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("Candidate destination escapes asset store")
        with self.edit() as catalog:
            existing = catalog["assets"].get(asset_id)
            if existing:
                if digest(self.path_for(existing).read_bytes()) != existing["sha256"]:
                    raise ValueError(f"Existing candidate corrupted: {asset_id}")
                return existing
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            record = {
                "asset_id": asset_id,
                "kind": "clip",
                "target_id": target_id,
                "file": f"candidates/{destination.name}",
                "sha256": sha,
                "metadata": json.loads(json.dumps(metadata or {})),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            catalog["assets"][asset_id] = record
            catalog["history"].append({"action": "import", "asset_id": asset_id, "at": record["created_at"]})
        return asset_id

    def approve(self, asset_id: str):
        with self.edit() as catalog:
            record = catalog["assets"][asset_id]
            kind_approvals = catalog["approvals"].setdefault(record["kind"], {})
            previous = kind_approvals.get(record["target_id"])
            if previous and previous != asset_id:
                raise ValueError("Target is locked. Explicitly unlock it with a reason before replacing its approval.")
            if digest(self.path_for(record).read_bytes()) != record["sha256"]:
                raise ValueError("Candidate checksum mismatch")
            kind_approvals[record["target_id"]] = asset_id
            catalog["history"].append({"action": "approve", "asset_id": asset_id, "at": datetime.now(timezone.utc).isoformat()})

    def unlock(self, kind: str, target_id: str, reason: str):
        if kind not in VALID_ASSET_KINDS or not reason.strip():
            raise ValueError("Unlock requires a valid kind and nonempty reason")
        with self.edit() as catalog:
            previous = catalog["approvals"].get(kind, {}).pop(target_id, None)
            catalog["history"].append({"action": "unlock", "kind": kind, "target_id": target_id,
                "previous": previous, "reason": reason, "at": datetime.now(timezone.utc).isoformat()})

    def approved(self, kind: str, target_id: str):
        if kind not in VALID_ASSET_KINDS:
            raise ValueError(f"Unsupported asset kind: {kind}")
        catalog = self.read()
        asset_id = catalog["approvals"].get(kind, {}).get(target_id)
        if not asset_id:
            return None
        record = catalog["assets"][asset_id]
        if record["kind"] != kind or record["target_id"] != target_id:
            raise ValueError("Approval target mismatch")
        path = self.path_for(record)
        if digest(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"Approved asset modified: {asset_id}")
        if kind in ("scene", "character"):
            read_image(path)
        elif not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Approved clip missing or empty: {asset_id}")
        return record, path
