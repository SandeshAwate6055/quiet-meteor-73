"""Adapt the Phase 2 notebook into a shared-code, resumable batch launcher.

This only writes notebook JSON. It does not execute cells or import GPU packages.
"""
import json
from pathlib import Path
import textwrap


def build():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "notebooks/phase2_image_generation.ipynb").read_text(encoding="utf-8"))
    old_intro = "".join(old["cells"][0]["source"])
    licenses = "Checkpoints and compatibility" + old_intro.split("Checkpoints and compatibility", 1)[1]
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "id": f"batch-{len(cells):02d}", "metadata": {},
                "source": textwrap.dedent(source).strip().splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", '''
    # AI Kids Story Studio - resumable batch image worker

    **Not GPU-executed during implementation.** The orchestration/state/import code is locally tested with synthetic fixtures.
    This launcher uses the SAME worker code shipped by your local production; no separate generation loop lives in the notebook.
    It cannot start a GPU runtime for you. Start on a CPU runtime for dry-run validation, then explicitly choose a GPU for generation.
    Upload the production `image_handoff/request_*.zip` ONCE. It contains the exact request, available references and worker source.
    Outputs, checkpoints, reference copies and exports live in Drive. Reopen the same package/result directory to resume.
    Default is `dry_run`: no model imports/downloads/inference and no result mutation.

    First run ONE smoke scene. Review its contact sheet, then set `MODE="full"`, `QUALITY_REVIEWED=True`, and increase
    `MAX_NEW_IMAGES` only enough for the remaining scenes. Full mode iterates every requested scene, not a hard-coded four.
    Unapproved references require explicit experimental preview consent; this NEVER approves them or the generated scenes locally.
    ''' + "\n\n" + licenses)
    add("code", '''
    # 1. Controls. Rerun this cell only after deliberately editing these values.
    MODE = "dry_run"  # dry_run, smoke, full
    MAX_NEW_IMAGES = 3  # per invocation, successful existing images do not count
    SMOKE_SCENE_ID = "challenge"
    RUN_GENERATION = False
    INSTALL_COLAB_DEPS = False
    LICENSES_REVIEWED = False
    ALLOW_UNAPPROVED_REFERENCE_PREVIEW = False
    QUALITY_REVIEWED = False  # Human smoke-quality acknowledgement, not production asset approval
    FAMILY = "sdxl"  # sd15 is the smaller explicit fallback; never silently substituted
    STEPS = 24
    BASE_SEED = 20261123
    IP_ADAPTER_SCALE = 0.55  # must be >0; character conditioning cannot be disabled
    PACKAGE_PATH = ""  # Set to the saved Drive ZIP path on subsequent sessions; otherwise upload once
    RESULT_BATCH_ID = ""  # blank => stable request-hash folder; use a NEW name for changed generation settings
    REFERENCE_BATCH_MANIFEST = ""  # optional existing Drive manifest (including the legacy two-reference batch)
    REFERENCE_IDS = {}  # only needed for ambiguous refs: {"hare": "exact candidate ID or sha256"}
    ''')
    add("code", '''
    # 2. Mount Drive, validate/extract the small handoff, load shared worker (NO MODELS).
    import json, hashlib, sys, uuid, zipfile
    from pathlib import Path, PurePosixPath
    from google.colab import drive, files
    drive.mount("/content/drive")
    SHARED = Path("/content/drive/MyDrive/AI_Kids_Story_Studio/batch_handoff")
    SHARED.mkdir(parents=True, exist_ok=True)
    if not PACKAGE_PATH:
        uploaded = files.upload()  # Choose the single production image_handoff/request_*.zip.
        if len(uploaded) != 1:
            raise ValueError("Upload exactly one handoff ZIP")
        payload = next(iter(uploaded.values()))
        package_hash = hashlib.sha256(payload).hexdigest()
        package = SHARED / (package_hash + ".zip")
        if not package.exists():
            with package.open("xb") as stream:
                stream.write(payload)
    else:
        package = Path(PACKAGE_PATH)
        package_hash = hashlib.sha256(package.read_bytes()).hexdigest()
    bundle = SHARED / package_hash
    with zipfile.ZipFile(package) as archive:
        if sum(m.file_size for m in archive.infolist()) > 128 * 1024 * 1024 or len(archive.infolist()) > 200:
            raise ValueError("Unexpectedly large handoff")
        index = json.loads(archive.read("bundle_index.json"))
        expected = set(index["files"]) | {"bundle_index.json"}
        if set(archive.namelist()) != expected:
            raise ValueError("Unexpected handoff files")
        for name in archive.namelist():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\\\" in name or ":" in name:
                raise ValueError("Unsafe ZIP path")
            payload = archive.read(name)
            if name != "bundle_index.json" and hashlib.sha256(payload).hexdigest() != index["files"][name]:
                raise ValueError("Handoff checksum mismatch")
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError("Existing persistent bundle was modified; refusing to overwrite")
            else:
                with target.open("xb") as stream:
                    stream.write(payload)
    previous_module = sys.modules.get("src.image_batch")
    if previous_module and not Path(previous_module.__file__).resolve().is_relative_to(bundle.resolve()):
        raise RuntimeError("A different bundle is already imported. Restart the session before loading new code.")
    sys.path.insert(0, str(bundle))
    from src.image_batch import BatchWorker, BatchSettings, reference_sources, contact_sheet
    request_hash = hashlib.sha256((bundle / "request.json").read_bytes()).hexdigest()
    if request_hash != index["request_sha256"]:
        raise ValueError("Production request hash mismatch")
    result_name = RESULT_BATCH_ID or ("request_" + request_hash[:16])
    if not result_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple alphanumeric result batch name")
    result_dir = SHARED / "results" / result_name
    print("Saved PACKAGE_PATH for resume:", package)
    print("Persistent result directory:", result_dir)
    ''')
    add("code", '''
    # 3. Preflight before installing/loading any GPU model. Read-only dry runs are safe.
    settings = BatchSettings(family=FAMILY, steps=STEPS, base_seed=BASE_SEED, adapter_scale=IP_ADAPTER_SCALE)
    references = reference_sources(bundle / "references.json")
    if REFERENCE_BATCH_MANIFEST:
        references += reference_sources(Path(REFERENCE_BATCH_MANIFEST))
    worker = BatchWorker(bundle / "request.json", result_dir, references=references, settings=settings)
    controls = dict(mode=MODE, smoke_scene=SMOKE_SCENE_ID, max_new_images=MAX_NEW_IMAGES,
        allow_unapproved_references=ALLOW_UNAPPROVED_REFERENCE_PREVIEW,
        quality_reviewed=QUALITY_REVIEWED, reference_ids=REFERENCE_IDS)
    preflight = worker.preflight(**controls)
    print(json.dumps({k: v for k, v in preflight.items() if k != "chosen_references"}, indent=2))
    print("Review count, reused images, warnings and blockers ABOVE before enabling GPU execution.")
    ''')
    add("code", '''
    # 4. Explicit execution. All initialization/adapter/generation errors are persisted by the worker.
    if MODE == "dry_run":
        outcome = worker.execute(lambda *args: (_ for _ in ()).throw(RuntimeError("Dry-run must not load a model")), **controls)
        print("DRY RUN ONLY:", json.dumps(outcome, indent=2))
    else:
        if preflight["blockers"]:
            raise RuntimeError("Resolve preflight blockers first: " + "; ".join(preflight["blockers"]))
        if not RUN_GENERATION or not LICENSES_REVIEWED:
            raise RuntimeError("Explicitly set RUN_GENERATION and LICENSES_REVIEWED after inspecting preflight/license terms.")
        if preflight["new_image_count"]:
            if not INSTALL_COLAB_DEPS:
                raise RuntimeError("Enable INSTALL_COLAB_DEPS for this Colab-only inference run.")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "diffusers==0.35.1",
                "transformers==4.56.2", "accelerate==1.10.1", "safetensors==0.6.2",
                "huggingface-hub==0.34.4", "Pillow>=10.4,<13"], check=True)
        from src.colab_images import ColabDiffusersBackend
        # Reuse one loaded pipeline between smoke and full within this session.
        # The persistent manifest still enables fresh-session resume after disconnect.
        if "_COLAB_BACKEND" not in globals():
            _COLAB_BACKEND = None
        def load_backend(settings, previous):
            global _COLAB_BACKEND
            if _COLAB_BACKEND is not None:
                if _COLAB_BACKEND.settings != settings:
                    raise RuntimeError("Changed model settings: restart the runtime and choose a new result batch.")
                if previous and previous.get("model_revision") != _COLAB_BACKEND.provenance["model_revision"]:
                    raise RuntimeError("Loaded model revision differs from the persistent batch; restart the session.")
                return _COLAB_BACKEND
            _COLAB_BACKEND = ColabDiffusersBackend(settings, previous)
            return _COLAB_BACKEND
        outcome = worker.execute(load_backend, **controls)
        print(json.dumps({k: outcome.get(k) for k in ("status", "completion", "errors")}, indent=2))
        if outcome["status"] == "failed":
            raise RuntimeError("Batch failed. Read its saved errors; successful images remain reusable. Export partial results if needed.")
    ''')
    add("code", '''
    # 5. One labeled review sheet for references + every requested scene, including missing/failed panels.
    from IPython.display import display
    from PIL import Image
    if (result_dir / "manifest.json").exists():
        review_path = contact_sheet(result_dir)
        display(Image.open(review_path))
        print("Review:", review_path)
    else:
        print("Dry run has no result images; no contact sheet generated.")
    ''')
    add("markdown", '''
    After the smoke scene: inspect it and the reference colors/identity. If unacceptable, STOP and export the partial batch.
    Do not use full mode to try to repair a bad smoke. A changed prompt/model/seed needs a fresh result batch.
    If acceptable, edit cell 1 to `MODE="full"`, `QUALITY_REVIEWED=True`, and set `MAX_NEW_IMAGES` to the
    preflight's remaining count (normally 4 with the existing two refs). Keep the saved `PACKAGE_PATH` and result folder,
    rerun cells 1-4, then review cell 5. Existing successful images and references are checksum-checked and reused.
    Completion covers ALL scene IDs in the original request; smoke success cannot mark a five-scene batch complete.

    A CUDA/adapter failure stops the invocation and is recorded without a text-only fallback or automatic costly retry.
    For OOM, export partial results; use a new batch with `FAMILY="sd15"` in a fresh runtime. The smaller model has lower
    quality and uses its matching SD 1.5 adapter. Identity consistency is not guaranteed for either model.
    ''')
    add("code", '''
    # 6. Export once per invocation; filenames are unique, all originals remain on Drive.
    import shutil
    if not (result_dir / "manifest.json").exists():
        print("Nothing generated; dry-run does not export a fake result.")
    else:
        export_root = SHARED / "exports"
        export_root.mkdir(exist_ok=True)
        archive_base = export_root / (result_dir.name + "_" + uuid.uuid4().hex[:12])
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=result_dir)
        print("Persistent export:", archive_path)
        files.download(archive_path)
        print("Disconnect/delete the GPU runtime now when finished, to stop idle credit use.")
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "colab": {"name": "phase2_batch_worker.ipynb"}}}
    path = root / "notebooks/phase2_batch_worker.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    build()
