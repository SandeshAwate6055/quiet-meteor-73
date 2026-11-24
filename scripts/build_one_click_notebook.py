"""Build notebooks/one_click_generation.ipynb and its pinned one-click handoff ZIP.

Writes notebook JSON only; never imports GPU packages or executes model inference.
"""
import base64
import json
from pathlib import Path
import textwrap

from src.one_click import DEFAULT_PACKAGE_NAME, prepare_one_click_handoff
from src.production_state import file_hash

ROOT = Path(__file__).resolve().parents[1]
REPO_RAW_BASE = "https://raw.githubusercontent.com/SandeshAwate6055/quiet-meteor-73/master"


def build():
    handoff = prepare_one_click_handoff(ROOT / "examples", name=DEFAULT_PACKAGE_NAME, overwrite=True)
    package_path = Path(handoff["package_path"])
    package_sha256 = file_hash(package_path)
    package_b64 = base64.b64encode(package_path.read_bytes()).decode("ascii")
    profile = json.loads((ROOT / "configs/one_click_generation.json").read_text(encoding="utf-8"))

    cells = []

    def add(kind, source):
        cell = {
            "cell_type": kind,
            "id": f"one-click-{len(cells):02d}",
            "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", f"""
    # AI Kids Story Studio — One-Click Character Reference Generation (v4)

    **How to run (One Click):**
    1. In Google Colab, select **Runtime > Change runtime type > L4 or A100 GPU** (with **High-RAM** enabled).
    2. Click **Runtime > Run all**.
    3. Wait for automatic hardware check, dependency setup, generation of the **two** character reference images (`hare` and `tortoise`), visual preview, and automatic result ZIP download.

    ---
    ### Storage & Runtime Notes
    - **Temporary storage by default:** Results, caches, and exports are stored in `/content/story_one_click` inside the Colab VM and **are permanently lost when the Colab runtime disconnects**. The result ZIP download triggers automatically at the end of execution.
    - **Optional persistent storage / resume:** No Google Drive authentication is required. If you want persistent Drive storage or cross-session resume, set `USE_GOOGLE_DRIVE = True` (or provide a custom `PERSISTENT_DIR`) in Cell 1 before running.
    - **Strict budget & cost protection:**
      - Generates **at most 2 character reference images** (`hare` and `tortoise`) at `1024x1024` in 4 distilled steps.
      - Zero scene generation, zero animation, zero automatic GPU retries, and no repeat model loads within the same runtime.
      - Checks CUDA, BF16 capability, VRAM (>= {profile['min_vram_gib']} GiB), host RAM (>= {profile['min_ram_gib']} GiB), and free disk (>= {profile['min_disk_gib']} GiB) **before** installing packages or downloading weights.
    - **Model & License:**
      - Model: [`{profile['model_id']}`](https://huggingface.co/{profile['model_id']}) (pinned commit `{profile['model_revision']}`).
      - Published License: **{profile['model_license']}** ([View LICENSE.md]({profile['license_url']})). Ungated public weights; no API key or Hugging Face login required.
      - Legacy SDXL (`src/colab_images.py`) remains preserved in the shared worker for fallback/scene workflows.
    - **Candidate status:** Generated images are **unapproved candidates** until you inspect the downloaded result ZIP and explicitly approve them locally.
    """)

    add("code", f"""
    # 1. Optional persistent storage settings (Defaults require ZERO edits: temporary Colab storage, full auto-run).
    USE_GOOGLE_DRIVE = False  # Set True ONLY if you want optional Google Drive persistence/resume.
    PERSISTENT_DIR = ""       # Optional custom directory path; leave blank for default /content/story_one_click.
    RESULT_BATCH_ID = ""      # Optional batch folder name for resuming an interrupted persistent batch.

    EXPECTED_PACKAGE_SHA256 = "{package_sha256}"
    EXPECTED_REQUEST_SHA256 = "{handoff['request_sha256']}"
    EXPECTED_PROFILE_SHA256 = "{handoff['profile_sha256']}"
    REMOTE_PACKAGE_URL = "{REPO_RAW_BASE}/examples/{DEFAULT_PACKAGE_NAME}.zip"
    MIN_VRAM_GIB = {profile['min_vram_gib']}
    MIN_RAM_GIB = {profile['min_ram_gib']}
    MIN_DISK_GIB = {profile['min_disk_gib']}
    """)

    add("code", """
    # 2. Immediate GPU, VRAM, host RAM and disk preflight BEFORE any dependency install or model download.
    from pathlib import Path
    import shutil
    import torch

    if USE_GOOGLE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        WORK_ROOT = Path(PERSISTENT_DIR or "/content/drive/MyDrive/AI_Kids_Story_Studio/one_click_v4")
        print("Persistent Google Drive storage enabled:", WORK_ROOT)
    else:
        WORK_ROOT = Path(PERSISTENT_DIR or "/content/story_one_click")
        print("Using temporary Colab storage:", WORK_ROOT)
        print("NOTE: Temporary storage is lost when the Colab runtime disconnects. Keep the auto-downloaded ZIP.")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    cuda_ok = torch.cuda.is_available()
    bf16_ok = bool(cuda_ok and torch.cuda.get_device_capability(0)[0] >= 8)
    vram_gib = (torch.cuda.get_device_properties(0).total_memory / 2**30) if cuda_ok else 0.0
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "NONE"
    meminfo = Path("/proc/meminfo")
    ram_gib = (
        int(next(line.split()[1] for line in meminfo.read_text().splitlines() if line.startswith("MemTotal:"))) / 1024**2
        if meminfo.exists() else 0.0
    )
    disk_gib = shutil.disk_usage(WORK_ROOT).free / 2**30

    print(f"Detected GPU: {gpu_name} | CUDA: {cuda_ok} | BF16: {bf16_ok} | VRAM: {vram_gib:.2f} GiB | Host RAM: {ram_gib:.2f} GiB | Free Disk: {disk_gib:.2f} GiB")
    instruction = "Select Runtime > Change runtime type > L4 or A100 GPU with High-RAM, then click Runtime > Run all. T4/CPU are not supported by this BF16 profile."
    if not cuda_ok or not bf16_ok or vram_gib < MIN_VRAM_GIB:
        raise RuntimeError(f"No compatible CUDA/BF16 GPU (need >= {MIN_VRAM_GIB} GiB VRAM, got {vram_gib:.2f} GiB on {gpu_name}). " + instruction)
    if ram_gib < MIN_RAM_GIB:
        raise RuntimeError(f"Insufficient host RAM (need >= {MIN_RAM_GIB} GiB for unquantized CPU offload, got {ram_gib:.2f} GiB). " + instruction)
    if disk_gib < MIN_DISK_GIB:
        raise RuntimeError(f"Insufficient free disk space (need >= {MIN_DISK_GIB} GiB, got {disk_gib:.2f} GiB). No packages or weights were downloaded.")
    """)

    add("code", f"""
    # 3. Automatically retrieve version-matched generation package, verify SHA256 integrity, and load shared worker.
    import base64, hashlib, json, sys, urllib.request
    from pathlib import Path

    EMBEDDED_PACKAGE_B64 = "{package_b64}"
    package_path = WORK_ROOT / "{DEFAULT_PACKAGE_NAME}_{package_sha256[:12]}.zip"

    if not package_path.exists() or hashlib.sha256(package_path.read_bytes()).hexdigest() != EXPECTED_PACKAGE_SHA256:
        payload = None
        try:
            with urllib.request.urlopen(REMOTE_PACKAGE_URL, timeout=15) as resp:
                candidate = resp.read()
            if hashlib.sha256(candidate).hexdigest() == EXPECTED_PACKAGE_SHA256:
                payload = candidate
                print("Retrieved version-matched handoff package from GitHub.")
        except Exception as exc:
            print("GitHub raw fetch unavailable or unverified, using embedded version-matched package:", exc)
        if payload is None:
            payload = base64.b64decode(EMBEDDED_PACKAGE_B64.encode("ascii"), validate=True)
            print("Loaded embedded version-matched handoff package.")
        if hashlib.sha256(payload).hexdigest() != EXPECTED_PACKAGE_SHA256:
            raise ValueError("Handoff package SHA256 integrity check failed!")
        package_path.write_bytes(payload)

    bootstrap_dir = WORK_ROOT / "bundles" / EXPECTED_PACKAGE_SHA256
    import zipfile
    with zipfile.ZipFile(package_path) as zf:
        if sum(m.file_size for m in zf.infolist()) > 128 * 1024 * 1024 or len(zf.infolist()) > 200:
            raise ValueError("Unexpectedly large handoff archive")
        index = json.loads(zf.read("bundle_index.json"))
        for mod in ("__init__.py", "asset_store.py", "characters.py", "schema.py", "production_state.py",
                    "image_conditioning.py", "image_batch.py", "colab_images.py", "modern_images.py", "one_click.py"):
            member = f"src/{{mod}}"
            data = zf.read(member)
            if hashlib.sha256(data).hexdigest() != index["files"][member]:
                raise ValueError(f"Bootstrap module checksum mismatch: {{member}}")
            target = bootstrap_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)

    previous_module = sys.modules.get("src.one_click")
    if previous_module and not Path(previous_module.__file__).resolve().is_relative_to(bootstrap_dir.resolve()):
        for name in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules.pop(name, None)
    if str(bootstrap_dir) not in sys.path:
        sys.path.insert(0, str(bootstrap_dir))

    from src.one_click import extract_and_verify_bundle
    from src.modern_images import check_hardware

    bundle_info = extract_and_verify_bundle(
        package_path,
        WORK_ROOT / "bundles",
        expected_package_sha256=EXPECTED_PACKAGE_SHA256,
    )
    BUNDLE_DIR = bundle_info["bundle_dir"]
    PROFILE = bundle_info["profile"]
    check_hardware(PROFILE, cuda=cuda_ok, bf16=bf16_ok, vram_gib=vram_gib, ram_gib=ram_gib, disk_gib=disk_gib)
    print("Verified bundle integrity:", BUNDLE_DIR)
    print("Model:", PROFILE["model_id"], "| Revision:", PROFILE["model_revision"], "| License:", PROFILE["model_license"])
    """)

    add("code", """
    # 4. Install missing Colab dependencies once (cached across reruns in the same runtime).
    import importlib.metadata, subprocess, sys

    req_file = BUNDLE_DIR / "requirements-one-click.txt"
    required_pins = {}
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            pkg, ver = line.split("==", 1)
            required_pins[pkg.strip()] = ver.strip()

    missing = []
    for pkg, ver in required_pins.items():
        try:
            if importlib.metadata.version(pkg) != ver:
                missing.append(f"{pkg}=={ver}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{pkg}=={ver}")

    if missing:
        print("Installing pinned one-click dependencies:", missing)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
        importlib.invalidate_caches()
    else:
        print("All pinned dependencies already installed.")
    """)

    add("code", """
    # 5. Load model once, generate the 2 character reference candidates (Hare & Tortoise), display & auto-download ZIP.
    import json
    from IPython.display import display
    from PIL import Image
    from google.colab import files
    from src.modern_images import FluxKleinBackend
    from src.one_click import run_one_click_batch

    if "_ONE_CLICK_BACKEND" not in globals():
        _ONE_CLICK_BACKEND = None

    CACHE_DIR = WORK_ROOT / "model_cache"

    def get_backend(worker_settings, previous):
        global _ONE_CLICK_BACKEND
        if _ONE_CLICK_BACKEND is not None:
            if _ONE_CLICK_BACKEND.settings != worker_settings:
                raise RuntimeError("Worker settings changed; restart runtime before switching settings.")
            return _ONE_CLICK_BACKEND
        _ONE_CLICK_BACKEND = FluxKleinBackend(worker_settings, PROFILE, CACHE_DIR, previous)
        return _ONE_CLICK_BACKEND

    run_result = run_one_click_batch(
        BUNDLE_DIR,
        WORK_ROOT,
        backend_factory=get_backend,
        cache_dir=CACHE_DIR,
        result_batch_id=RESULT_BATCH_ID or None,
    )

    print("Batch completion summary:", json.dumps(run_result["validation"], indent=2))
    print("Contact sheet:", run_result["contact_sheet_path"])
    with Image.open(run_result["contact_sheet_path"]) as sheet_img:
        sheet_img.load()
        display(sheet_img)

    for asset in run_result["outcome"]["assets"]:
        img_path = run_result["result_dir"] / asset["file"]
        print(f"Candidate {asset['kind']}/{asset['target_id']} | seed={asset['metadata']['seed']} | sha256={asset['sha256'][:16]}:")
        with Image.open(img_path) as cand_img:
            cand_img.load()
            display(cand_img)

    print("Exported result ZIP:", run_result["archive_path"], "| SHA256:", run_result["archive_sha256"])
    files.download(str(run_result["archive_path"]))
    print("Download triggered! Temporary Colab files are lost when the runtime disconnects. Disconnect/delete the GPU runtime once your ZIP download finishes.")
    """)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "one_click_generation.ipynb", "gpuType": "L4", "machine_shape": "hm"},
        },
    }
    notebook_path = ROOT / "notebooks/one_click_generation.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return {
        "notebook_path": str(notebook_path),
        "package_path": str(package_path),
        "package_sha256": package_sha256,
        "request_sha256": handoff["request_sha256"],
        "profile_sha256": handoff["profile_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
