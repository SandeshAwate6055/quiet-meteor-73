"""Build notebooks/one_click_generation.ipynb, notebooks/one_click_challenge_scene.ipynb,
notebooks/one_click_remaining_scenes.ipynb, and their pinned one-click handoff ZIPs.

Writes notebook JSON only; never imports GPU packages or executes model inference.
"""
import json
from pathlib import Path
import subprocess
import textwrap

from src.one_click import (
    DEFAULT_CHALLENGE_PACKAGE_NAME,
    DEFAULT_EPISODE_CORRECTIONS_PACKAGE_NAME,
    DEFAULT_EPISODE_PACKAGE_NAME,
    DEFAULT_PACKAGE_NAME,
    DEFAULT_REMAINING_SCENES_PACKAGE_NAME,
    prepare_one_click_challenge_handoff,
    prepare_one_click_episode_handoff,
    prepare_one_click_handoff,
    prepare_one_click_remaining_scenes_handoff,
)
from src.one_click_video import (
    DEFAULT_I2V_TEST_PACKAGE_NAME,
    prepare_one_click_i2v_handoff,
)
from src.production_state import file_hash

ROOT = Path(__file__).resolve().parents[1]
REPO_SLUG = "SandeshAwate6055/quiet-meteor-73"
REPO_RAW_BASE = f"https://raw.githubusercontent.com/{REPO_SLUG}"


def _package_commit(package_rel_path):
    """Resolve the immutable git commit where package_rel_path was last modified, or HEAD."""
    for args in (
        ["git", "log", "-n", "1", "--format=%H", "--", package_rel_path],
        ["git", "rev-parse", "HEAD"],
    ):
        proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
        sha = proc.stdout.strip()
        if proc.returncode == 0 and len(sha) == 40:
            return sha
    return "master"


def _write_notebook(notebook_filename, title, description_lines, package_name, handoff, profile, cell5_header):
    package_path = Path(handoff["package_path"])
    package_sha256 = file_hash(package_path)
    rel_package = f"examples/{package_name}.zip"
    commit_sha = _package_commit(rel_package)
    immutable_url = f"{REPO_RAW_BASE}/{commit_sha}/{rel_package}"
    branch_url = f"{REPO_RAW_BASE}/master/{rel_package}"

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
    # {title}

    **How to run (One Click):**
    1. In Google Colab, select **Runtime > Change runtime type > L4 or A100 GPU** (with **High-RAM** enabled).
    2. Click **Runtime > Run all**.
    3. {description_lines['step3']}

    ---
    ### Storage & Runtime Notes
    - **Temporary storage by default:** Results, caches, and exports are stored in `/content/story_one_click` inside the Colab VM and **are permanently lost when the Colab runtime disconnects**. The result ZIP download triggers automatically at the end of execution.
    - **Optional persistent storage / resume:** No Google Drive authentication is required. If you want persistent Drive storage or cross-session resume, set `USE_GOOGLE_DRIVE = True` (or provide a custom `PERSISTENT_DIR`) in Cell 1 before running.
    - **Strict budget & cost protection:**
      - {description_lines['budget']}
      - Zero animation, zero automatic GPU retries, and no repeat model loads within the same runtime.
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
    REMOTE_PACKAGE_URLS = (
        "{immutable_url}",
        "{branch_url}",
    )
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
    import hashlib, json, sys, urllib.request, zipfile
    from pathlib import Path

    package_path = WORK_ROOT / "{package_name}_{package_sha256[:12]}.zip"

    if not package_path.exists() or hashlib.sha256(package_path.read_bytes()).hexdigest() != EXPECTED_PACKAGE_SHA256:
        payload = None
        fetch_errors = []
        for url in REMOTE_PACKAGE_URLS:
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    candidate = resp.read()
                actual_sha = hashlib.sha256(candidate).hexdigest()
                if actual_sha != EXPECTED_PACKAGE_SHA256:
                    fetch_errors.append(f"{{url}}: SHA256 mismatch (expected {{EXPECTED_PACKAGE_SHA256}}, got {{actual_sha}})")
                    continue
                payload = candidate
                print("Retrieved and SHA256-verified handoff package from:", url)
                break
            except Exception as exc:
                fetch_errors.append(f"{{url}}: {{exc}}")
        if payload is None:
            raise RuntimeError(
                "Could not download a SHA256-verified handoff package; refusing to run with an unverified or missing bundle. Details: "
                + "; ".join(fetch_errors)
            )
        package_path.write_bytes(payload)

    bootstrap_dir = WORK_ROOT / "bundles" / EXPECTED_PACKAGE_SHA256
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

    add("code", f"""
    # 5. {cell5_header}
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
        allow_partial=True,
    )

    print("Batch completion summary:", json.dumps(run_result["validation"], indent=2))
    print("Contact sheet:", run_result["contact_sheet_path"])
    with Image.open(run_result["contact_sheet_path"]) as sheet_img:
        sheet_img.load()
        display(sheet_img)

    for asset in run_result["outcome"]["assets"]:
        img_path = run_result["result_dir"] / asset["file"]
        status_tag = "REUSED APPROVED REFERENCE" if asset["metadata"].get("reused_reference") else f"GENERATED CANDIDATE (seed={{asset['metadata'].get('seed')}})"
        print(f"{{asset['kind']}}/{{asset['target_id']}} | {{status_tag}} | sha256={{asset['sha256'][:16]}}:")
        if not asset["metadata"].get("reused_reference"):
            with Image.open(img_path) as cand_img:
                cand_img.load()
                display(cand_img)

    print("Exported result ZIP:", run_result["archive_path"], "| SHA256:", run_result["archive_sha256"])
    files.download(str(run_result["archive_path"]))
    if run_result["outcome"].get("status") != "complete":
        raise RuntimeError(
            f"Batch did not complete ({{run_result['outcome'].get('status')}}); partial export downloaded from {{run_result['archive_path']}}: {{run_result['outcome'].get('errors')}}"
        )
    print("Download triggered! Temporary Colab files are lost when the runtime disconnects. Disconnect/delete the GPU runtime once your ZIP download finishes.")
    """)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": notebook_filename, "gpuType": "L4", "machine_shape": "hm"},
        },
    }
    notebook_path = ROOT / "notebooks" / notebook_filename
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return {
        "notebook_path": str(notebook_path),
        "package_path": str(package_path),
        "package_sha256": package_sha256,
        "request_sha256": handoff["request_sha256"],
        "profile_sha256": handoff["profile_sha256"],
    }


def build_references():
    handoff = prepare_one_click_handoff(ROOT / "examples", name=DEFAULT_PACKAGE_NAME, overwrite=True)
    profile = json.loads((ROOT / "configs/one_click_generation.json").read_text(encoding="utf-8"))
    return _write_notebook(
        "one_click_generation.ipynb",
        "AI Kids Story Studio — One-Click Character Reference Generation (v4)",
        {
            "step3": "Wait for automatic hardware check, dependency setup, generation of the **two** character reference images (`hare` and `tortoise`), visual preview, and automatic result ZIP download.",
            "budget": "Generates **at most 2 character reference images** (`hare` and `tortoise`) at `1024x1024` in 4 distilled steps. Zero scene generation.",
        },
        DEFAULT_PACKAGE_NAME,
        handoff,
        profile,
        "Load model once, generate the 2 character reference candidates (Hare & Tortoise), display & auto-download ZIP.",
    )


def build_challenge():
    handoff = prepare_one_click_challenge_handoff(ROOT / "examples", name=DEFAULT_CHALLENGE_PACKAGE_NAME, overwrite=True)
    profile = json.loads((ROOT / "configs/one_click_challenge_scene.json").read_text(encoding="utf-8"))
    return _write_notebook(
        "one_click_challenge_scene.ipynb",
        "AI Kids Story Studio — One-Click First Scene Generation: `challenge` (v4)",
        {
            "step3": "Wait for automatic hardware check, dependency setup, reuse of the **approved** `hare` and `tortoise` references, generation of **only** the `challenge` scene (`1024x576`), visual preview, and automatic result ZIP download.",
            "budget": "Reuses the 2 approved character references (`hare` and `tortoise`, 0 reference regenerations) and generates **at most 1 scene image** (`challenge`) at `1024x576` in 4 distilled steps. Remaining scenes are not scheduled.",
        },
        DEFAULT_CHALLENGE_PACKAGE_NAME,
        handoff,
        profile,
        "Load model once, reuse approved Hare & Tortoise references, generate ONLY the 'challenge' scene, display & auto-download ZIP.",
    )


def build_remaining_scenes():
    handoff = prepare_one_click_remaining_scenes_handoff(
        ROOT / "examples", name=DEFAULT_REMAINING_SCENES_PACKAGE_NAME, overwrite=True
    )
    profile = json.loads((ROOT / "configs/one_click_remaining_scenes.json").read_text(encoding="utf-8"))
    return _write_notebook(
        "one_click_remaining_scenes.ipynb",
        "AI Kids Story Studio — One-Click Remaining Four Story Scenes (`racing_ahead`, `nap`, `steady_steps`, `finish`) (v4)",
        {
            "step3": "Wait for automatic hardware check, dependency setup, reuse of the **approved** `hare` and `tortoise` references, generation of the **four remaining scenes** (`racing_ahead`, `nap`, `steady_steps`, `finish` at `1024x576`), visual preview, and automatic result ZIP download.",
            "budget": "Reuses the 2 approved character references (`hare` and `tortoise`, 0 reference regenerations), preserves the already approved `challenge` scene (0 `challenge` regenerations), and generates **at most 4 new scene images** (`racing_ahead`, `nap`, `steady_steps`, `finish`) at `1024x576` in 4 distilled steps.",
        },
        DEFAULT_REMAINING_SCENES_PACKAGE_NAME,
        handoff,
        profile,
        "Load model once, reuse approved Hare & Tortoise references, generate the 4 remaining scenes (preserving approved 'challenge'), display & auto-download ZIP.",
    )


def build_episode_scenes():
    handoff = prepare_one_click_episode_handoff(
        ROOT / "scripts/episode_01_golden_acorn.json",
        ROOT / "examples",
        name=DEFAULT_EPISODE_PACKAGE_NAME,
        overwrite=True,
    )
    profile = {
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "model_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "model_license": "apache-2.0",
        "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md",
        "min_vram_gib": 22,
        "min_ram_gib": 24,
        "min_disk_gib": 24,
    }
    count = len(handoff["scene_ids"])
    reused = len(handoff["reused_shot_ids"])
    return _write_notebook(
        "one_click_episode_scenes.ipynb",
        f"AI Kids Story Studio — One-Click Episode 1 Missing Scenes ({count} New Shots, {reused} Reused) (v4)",
        {
            "step3": f"Wait for automatic hardware check, dependency setup, reuse of the **approved** `hare` and `tortoise` references, generation of the **{count} missing episode shots** at `1024x576`, contact sheet preview, and automatic result ZIP download.",
            "budget": f"Reuses the 2 approved character references (`hare` and `tortoise`, 0 reference regenerations), preserves all {reused} already-approved scenes (`{', '.join(handoff['excluded_approved_scene_ids'])}`), and generates **at most {count} missing scene images** at `1024x576` in 4 distilled steps. Supports partial export & resume.",
        },
        DEFAULT_EPISODE_PACKAGE_NAME,
        handoff,
        profile,
        f"Load model once, reuse approved Hare & Tortoise references, generate the {count} missing episode shots (preserving {reused} approved scenes), display contact sheet & auto-download ZIP.",
    )


def build_episode_corrections():
    handoff = prepare_one_click_episode_handoff(
        ROOT / "scripts/episode_01_golden_acorn.json",
        ROOT / "examples",
        name=DEFAULT_EPISODE_CORRECTIONS_PACKAGE_NAME,
        profile_path=ROOT / "configs/one_click_episode_corrections.json",
        overwrite=True,
    )
    profile = json.loads((ROOT / "configs/one_click_episode_corrections.json").read_text(encoding="utf-8"))
    count = len(handoff["scene_ids"])
    reused = len(handoff["reused_shot_ids"])
    return _write_notebook(
        "one_click_episode_corrections.ipynb",
        f"AI Kids Story Studio — One-Click Episode 1 Corrections ({count} Targeted Shots, {reused} Reused) (v4)",
        {
            "step3": f"Wait for automatic hardware check, dependency setup, reuse of the **approved** `hare` and `tortoise` references, regeneration of **only the {count} targeted correction shots** (`{', '.join(handoff['scene_ids'])}`) at `1024x576`, contact sheet preview, and automatic result ZIP download.",
            "budget": f"Reuses the 2 approved character references (`hare` and `tortoise`, 0 reference regenerations), preserves all {reused} already-approved scenes, and generates **at most {count} targeted correction images** (`{', '.join(handoff['scene_ids'])}`) at `1024x576` in 4 distilled steps.",
        },
        DEFAULT_EPISODE_CORRECTIONS_PACKAGE_NAME,
        handoff,
        profile,
        f"Load model once, reuse approved Hare & Tortoise references, generate ONLY the {count} correction shots ({', '.join(handoff['scene_ids'])}), display contact sheet & auto-download ZIP.",
    )


def _write_i2v_notebook(notebook_filename, title, package_name, handoff, profile):
    package_path = Path(handoff["package_path"])
    package_sha256 = handoff["package_sha256"]
    rel_package = f"examples/{package_name}.zip"
    commit = _package_commit(rel_package)
    immutable_url = f"{REPO_RAW_BASE}/{commit}/{rel_package}"
    branch_url = f"{REPO_RAW_BASE}/master/{rel_package}"

    cells = []

    def add(kind, source):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": [line + "\n" for line in textwrap.dedent(source).strip("\n").splitlines()],
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    add("markdown", f"""
    # {title}

    **How to run (One Click):**
    1. In Google Colab, select **Runtime > Change runtime type > L4 or A100 GPU** (with **High-RAM** enabled).
    2. Click **Runtime > Run all**.
    3. Wait for automatic hardware check, dependency setup, generation of **1 test I2V MP4 clip** (`hare_sprints`, `832x480`, `81` frames @ `16` fps), and automatic result ZIP download.

    ---
    ### Budget & License Guardrails
    - **Strict 1-clip test budget:** Generates **at most {profile['max_new_clips']} clip** (`{', '.join(handoff['scene_ids'])}`), with zero automatic retries and no repeat model loads.
    - **Model & License:** [`{profile['model_id']}`](https://huggingface.co/{profile['model_id']}) (pinned commit `{profile['model_revision']}`), **{profile['model_license']}** ([View LICENSE.txt]({profile['license_url']})). Ungated public weights.
    - **Candidate status:** The generated MP4 is an **unapproved candidate** until inspected and explicitly approved locally via `AssetStore`.
    """)

    add("code", f"""
    # 1. Optional persistent storage settings (Defaults require ZERO edits: temporary Colab storage, full auto-run).
    USE_GOOGLE_DRIVE = False
    PERSISTENT_DIR = ""

    EXPECTED_PACKAGE_SHA256 = "{package_sha256}"
    EXPECTED_REQUEST_SHA256 = "{handoff['request_sha256']}"
    EXPECTED_PROFILE_SHA256 = "{handoff['profile_sha256']}"
    REMOTE_PACKAGE_URLS = (
        "{immutable_url}",
        "{branch_url}",
    )
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
        WORK_ROOT = Path(PERSISTENT_DIR or "/content/drive/MyDrive/AI_Kids_Story_Studio/one_click_i2v")
    else:
        WORK_ROOT = Path(PERSISTENT_DIR or "/content/story_one_click_i2v")
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
    instruction = "Select Runtime > Change runtime type > L4 or A100 GPU with High-RAM, then click Runtime > Run all."
    if not cuda_ok or not bf16_ok or vram_gib < MIN_VRAM_GIB:
        raise RuntimeError(f"No compatible CUDA/BF16 GPU (need >= {MIN_VRAM_GIB} GiB VRAM, got {vram_gib:.2f} GiB on {gpu_name}). " + instruction)
    if ram_gib < MIN_RAM_GIB:
        raise RuntimeError(f"Insufficient host RAM (need >= {MIN_RAM_GIB} GiB, got {ram_gib:.2f} GiB). " + instruction)
    if disk_gib < MIN_DISK_GIB:
        raise RuntimeError(f"Insufficient free disk space (need >= {MIN_DISK_GIB} GiB, got {disk_gib:.2f} GiB).")
    """)

    add("code", f"""
    # 3. Download and SHA256-verify the I2V handoff package, extract worker modules, and check hardware profile.
    import hashlib, json, sys, urllib.request, zipfile
    from pathlib import Path

    package_path = WORK_ROOT / "{package_name}_{package_sha256[:12]}.zip"
    if not package_path.exists() or hashlib.sha256(package_path.read_bytes()).hexdigest() != EXPECTED_PACKAGE_SHA256:
        payload = None
        fetch_errors = []
        for url in REMOTE_PACKAGE_URLS:
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    candidate = resp.read()
                actual_sha = hashlib.sha256(candidate).hexdigest()
                if actual_sha != EXPECTED_PACKAGE_SHA256:
                    fetch_errors.append(f"{{url}}: SHA256 mismatch (expected {{EXPECTED_PACKAGE_SHA256}}, got {{actual_sha}})")
                    continue
                payload = candidate
                print("Retrieved and SHA256-verified I2V handoff package from:", url)
                break
            except Exception as exc:
                fetch_errors.append(f"{{url}}: {{exc}}")
        if payload is None:
            raise RuntimeError("Could not download SHA256-verified I2V package: " + "; ".join(fetch_errors))
        package_path.write_bytes(payload)

    bootstrap_dir = WORK_ROOT / "bundles" / EXPECTED_PACKAGE_SHA256
    with zipfile.ZipFile(package_path) as zf:
        index = json.loads(zf.read("bundle_index.json"))
        for mod in ("__init__.py", "asset_store.py", "characters.py", "schema.py",
                    "production_state.py", "modern_video.py", "one_click_video.py"):
            member = f"src/{{mod}}"
            data = zf.read(member)
            if hashlib.sha256(data).hexdigest() != index["files"][member]:
                raise ValueError(f"Bootstrap module checksum mismatch: {{member}}")
            target = bootstrap_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(data)

    if str(bootstrap_dir) not in sys.path:
        sys.path.insert(0, str(bootstrap_dir))

    from src.one_click_video import extract_and_verify_i2v_bundle
    from src.modern_video import check_i2v_hardware

    bundle_info = extract_and_verify_i2v_bundle(
        package_path, WORK_ROOT / "bundles", expected_package_sha256=EXPECTED_PACKAGE_SHA256
    )
    BUNDLE_DIR = bundle_info["bundle_dir"]
    PROFILE = bundle_info["profile"]
    check_i2v_hardware(PROFILE, cuda=cuda_ok, bf16=bf16_ok, vram_gib=vram_gib, ram_gib=ram_gib, disk_gib=disk_gib)
    print("Verified I2V bundle:", BUNDLE_DIR)
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
    for extra in ("ftfy", "imageio", "imageio-ffmpeg"):
        try:
            importlib.metadata.version(extra)
        except importlib.metadata.PackageNotFoundError:
            missing.append(extra)

    if missing:
        print("Installing I2V dependencies:", missing)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
        importlib.invalidate_caches()
    else:
        print("All I2V dependencies already installed.")
    """)

    add("code", """
    # 5. Load Wan2.2-TI2V-5B once, animate ONLY 'hare_sprints' (81 frames @ 16 fps), preview & auto-download ZIP.
    import json
    from IPython.display import Video, display
    from google.colab import files
    from src.modern_video import WanI2VBackend
    from src.one_click_video import run_one_click_i2v_batch

    if "_ONE_CLICK_I2V_BACKEND" not in globals():
        _ONE_CLICK_I2V_BACKEND = None

    CACHE_DIR = WORK_ROOT / "model_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_backend(worker_settings, previous):
        global _ONE_CLICK_I2V_BACKEND
        if _ONE_CLICK_I2V_BACKEND is not None:
            if _ONE_CLICK_I2V_BACKEND.settings != worker_settings:
                raise RuntimeError("I2V worker settings changed; restart runtime.")
            return _ONE_CLICK_I2V_BACKEND
        _ONE_CLICK_I2V_BACKEND = WanI2VBackend(worker_settings, PROFILE, CACHE_DIR, previous)
        return _ONE_CLICK_I2V_BACKEND

    run_result = run_one_click_i2v_batch(BUNDLE_DIR, WORK_ROOT, backend_factory=get_backend, cache_dir=CACHE_DIR)
    print("I2V batch outcome:", json.dumps(run_result["outcome"], indent=2))
    for asset in run_result["outcome"]["assets"]:
        clip_path = run_result["result_dir"] / asset["file"]
        print(f"Generated {asset['kind']}/{asset['target_id']} | sha256={asset['sha256'][:16]}")
        display(Video(str(clip_path), embed=True))

    print("Exported result ZIP:", run_result["archive_path"], "| SHA256:", run_result["archive_sha256"])
    files.download(str(run_result["archive_path"]))
    """)

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": notebook_filename, "gpuType": "L4", "machine_shape": "hm"},
        },
    }
    notebook_path = ROOT / "notebooks" / notebook_filename
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return {
        "notebook_path": str(notebook_path),
        "package_path": str(package_path),
        "package_sha256": package_sha256,
        "request_sha256": handoff["request_sha256"],
        "profile_sha256": handoff["profile_sha256"],
    }


def build_i2v_test():
    handoff = prepare_one_click_i2v_handoff(ROOT / "examples", name=DEFAULT_I2V_TEST_PACKAGE_NAME, overwrite=True)
    profile = json.loads((ROOT / "configs/one_click_i2v_test.json").read_text(encoding="utf-8"))
    return _write_i2v_notebook(
        "one_click_i2v_test.ipynb",
        "AI Kids Story Studio — One-Click I2V Quality Test: `hare_sprints` (Wan2.2-TI2V-5B)",
        DEFAULT_I2V_TEST_PACKAGE_NAME,
        handoff,
        profile,
    )


def build():
    ref_result = build_references()
    corrections_result = build_episode_corrections()
    i2v_test_result = build_i2v_test()
    result = {
        **ref_result,
        "episode_corrections": corrections_result,
        "i2v_test": i2v_test_result,
    }
    if not (ROOT / "notebooks/one_click_episode_scenes.ipynb").exists():
        result["episode_scenes"] = build_episode_scenes()
    if not (ROOT / "notebooks/one_click_challenge_scene.ipynb").exists():
        result["challenge"] = build_challenge()
    if not (ROOT / "notebooks/one_click_remaining_scenes.ipynb").exists():
        result["remaining_scenes"] = build_remaining_scenes()
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
