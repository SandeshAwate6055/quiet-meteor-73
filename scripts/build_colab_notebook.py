"""Build a reviewable notebook artifact using only the Python standard library.

Running this script NEVER executes a notebook cell or downloads any package/model.
"""
import json
from pathlib import Path
import textwrap


def build():
    cells = []

    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)})

    def code(source):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": textwrap.dedent(source).strip().splitlines(True)})

    md('''
    # Phase 2: small-batch story illustration generation (Colab only)

    **Status: prepared locally; not executed on Colab.** No Windows ML installation is needed.
    Run individual cells in order, not Run all. Default scope: two character references and **one** scene (`challenge`).
    Later choose all four scenes: `challenge`, `racing_ahead`, `steady_steps`, `finish`.
    The fifth Phase 1 scene, `nap`, intentionally retains its Pillow fallback until generated separately.

    Before starting, export `colab_input.json` locally using `python -m src.assets export-request`.
    In Colab choose **Runtime > Change runtime type > Python 3 > GPU** (T4 is sufficient to attempt the low-memory workflow).
    Starting a GPU runtime can consume credits while idle; disconnect/delete it when finished. No paid inference APIs are called.

    Checkpoints and compatibility checked against their published model cards/docs:
    - [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0):
      [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md).
    - [SD 1.5 fallback](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5):
      CreativeML OpenRAIL-M (use restrictions apply; this is a community mirror, not affiliated with Runway).
    - [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter): Apache-2.0.
      SDXL uses `sdxl_models/ip-adapter_sdxl_vit-h.safetensors` plus `models/image_encoder` (ViT-H).
      SD 1.5 uses `models/ip-adapter_sd15.safetensors` with its matching ViT-H encoder.
    - [Diffusers 0.35.1 adapter documentation](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/ip_adapter),
      [memory guidance](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory).

    These are open weights with license conditions, not public-domain checkpoints. Read the linked terms before use/redistribution.
    IP-Adapter guides appearance; it does **not guarantee** animal identity, clothing, shell patterns or two-character separation.
    A combined reference board is a pragmatic starting point, not spatially controlled multi-character conditioning.
    Review every result; no output is approved automatically.
    ''')
    code('''
    # 1. Colab-only dependency install. Keep the runtime's preinstalled GPU PyTorch.
    import sys, subprocess
    try:
        import google.colab
    except ImportError as exc:
        raise RuntimeError("This notebook installs ML packages ONLY in Google Colab. Do not run on Windows.") from exc
    INSTALL_COLAB_DEPS = False  # Set True only when manually running this cell in Colab.
    if not INSTALL_COLAB_DEPS:
        raise RuntimeError("Set INSTALL_COLAB_DEPS=True in Colab to install the pinned inference libraries.")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
        "diffusers==0.35.1", "transformers==4.56.2", "accelerate==1.10.1",
        "safetensors==0.6.2", "huggingface-hub==0.34.4", "Pillow>=10.4,<13"], check=True)
    print("If Colab requests a session restart, restart it, then continue with cell 2.")
    ''')
    code('''
    # 2. Inspect hardware; no model download.
    import os, json, time, uuid, hashlib, base64, gc, shutil, platform
    from pathlib import Path
    from io import BytesIO
    from datetime import datetime, timezone
    import importlib.metadata
    import torch
    from PIL import Image, ImageOps
    from IPython.display import display
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU: change runtime type to GPU. CPU generation is intentionally disabled.")
    props = torch.cuda.get_device_properties(0)
    GPU_GIB = props.total_memory / 2**30
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    RAM_FREE_GIB = int(next(s for s in meminfo if s.startswith("MemAvailable:")).split()[1]) / 2**20
    print({"gpu": props.name, "vram_GiB": round(GPU_GIB, 2), "free_RAM_GiB": round(RAM_FREE_GIB, 2),
           "torch": torch.__version__, "cuda": torch.version.cuda})
    ''')
    md('''
    ## First quality test settings

    Leave the three-image limit and `SELECT_SCENES=["challenge"]` unchanged for the first test.
    Set `RUN_GENERATION=True` and `I_HAVE_READ_MODEL_LICENSES=True` only when ready to download weights and generate.
    `auto` attempts SDXL only with at least 14 GiB VRAM and 10 GiB free host RAM; this is a heuristic, not a fit guarantee.
    Both workflows use fp16, one image at a time, VAE slicing/tiling and CPU offload.
    Lower-memory GPUs use SD 1.5 at 512 pixels; below 8 GiB VRAM it uses slower sequential CPU offload.
    No refiner, LoRA, training, face recognition or video model is loaded.
    ''')
    code('''
    # 3. Explicit job scope. Rerunning this cell resets these flags; edit deliberately.
    RUN_GENERATION = False
    I_HAVE_READ_MODEL_LICENSES = False
    MODEL_FAMILY = "auto"  # auto, sdxl, sd15
    SELECT_SCENES = ["challenge"]
    GENERATE_REFERENCES = True  # False for scene-only regeneration using exported approved refs.
    MAX_NEW_IMAGES = 3
    BASE_SEED = 20261123
    STEPS = 24
    IP_ADAPTER_SCALE = 0.55
    USE_REFERENCE_CONDITIONING = True
    REFERENCES_OK = False  # Set True in cell 8 only AFTER inspecting the reference previews.
    if MODEL_FAMILY not in {"auto", "sdxl", "sd15"}:
        raise ValueError("Unknown model family")
    FAMILY = ("sdxl" if GPU_GIB >= 14 and RAM_FREE_GIB >= 10 else "sd15") if MODEL_FAMILY == "auto" else MODEL_FAMILY
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" if FAMILY == "sdxl" else "stable-diffusion-v1-5/stable-diffusion-v1-5"
    MODEL_LICENSE = "CreativeML Open RAIL++-M" if FAMILY == "sdxl" else "CreativeML OpenRAIL-M"
    REF_SIZE = (768, 768) if FAMILY == "sdxl" else (512, 512)
    SCENE_SIZE = (1024, 576) if FAMILY == "sdxl" else (768, 432)
    if GPU_GIB < 8:
        SCENE_SIZE = (512, 384)
    print({"family": FAMILY, "model": MODEL_ID, "reference_size": REF_SIZE, "scene_size": SCENE_SIZE,
           "new_image_limit": MAX_NEW_IMAGES, "selected_scenes": SELECT_SCENES})
    ''')
    code('''
    # 4. Mount persistent storage and upload the locally exported request JSON.
    from google.colab import drive, files
    drive.mount("/content/drive")
    uploaded = files.upload()  # Choose notebooks/colab_input.json from your laptop.
    names = [name for name in uploaded if name.endswith(".json")]
    if len(names) != 1:
        raise ValueError("Upload exactly one JSON request exported by src.assets")
    REQUEST_BYTES = uploaded[names[0]]
    request = json.loads(REQUEST_BYTES)
    if request.get("version") != 1:
        raise ValueError("Unsupported request version")
    scenes_by_id = {s["id"]: s for s in request["scenes"]}
    chars_by_id = {c["id"]: c for c in request["characters"]}
    if not SELECT_SCENES or len(set(SELECT_SCENES)) != len(SELECT_SCENES) or set(SELECT_SCENES) - set(scenes_by_id):
        raise ValueError("Selected scene IDs must be unique and present in the exported request")
    locked = set(SELECT_SCENES) & set(request.get("locked_scene_ids", []))
    if locked:
        raise ValueError(f"Scenes are approved/locked: {sorted(locked)}. Unlock locally with a reason, then export a fresh request.")
    RUN_ROOT = Path("/content/drive/MyDrive/AI_Kids_Story_Studio/image_batches")
    RUN_DIR = RUN_ROOT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    RUN_DIR.mkdir(parents=True, exist_ok=False)  # New batch only; no existing images overwritten.
    (RUN_DIR / "request.json").write_bytes(REQUEST_BYTES)
    manifest = {"version": 1, "status": "prepared", "assets": [], "errors": [],
        "created_at": datetime.now(timezone.utc).isoformat(), "gpu": props.name,
        "vram_GiB": GPU_GIB, "host_free_RAM_GiB": RAM_FREE_GIB,
        "model_id": MODEL_ID, "model_license": MODEL_LICENSE,
        "request_sha256": hashlib.sha256(REQUEST_BYTES).hexdigest(),
        "versions": {n: importlib.metadata.version(n) for n in ["diffusers", "transformers", "accelerate", "safetensors", "huggingface-hub", "Pillow"]},
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()}
    def save_manifest():
        temporary = RUN_DIR / "manifest.tmp.json"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(RUN_DIR / "manifest.json")
    save_manifest()
    refs = {}
    ref_sources = {}
    for identity, character in chars_by_id.items():
        encoded = character.get("approved_reference_png_base64")
        if encoded:
            payload = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(payload).hexdigest() != character["approved_reference_sha256"]:
                raise ValueError("Approved reference checksum mismatch")
            refs[identity] = Image.open(BytesIO(payload)).convert("RGB")
            relative = f"inputs/{identity}.png"
            (RUN_DIR / "inputs").mkdir(exist_ok=True)
            with (RUN_DIR / relative).open("xb") as stream:
                stream.write(payload)
            ref_sources[identity] = {"file": relative, "sha256": character["approved_reference_sha256"],
                                     "approved_asset_id": character["approved_asset_id"]}
    needed = set(c for sid in SELECT_SCENES for c in scenes_by_id[sid]["characters"])
    # The sample first test generates both references. Existing approved references are always reused.
    new_ref_ids = [c for c in chars_by_id if c not in refs] if GENERATE_REFERENCES else []
    if needed - set(refs) - set(new_ref_ids) and USE_REFERENCE_CONDITIONING:
        raise ValueError("Missing references. Export approved references or enable GENERATE_REFERENCES.")
    if len(new_ref_ids) + len(SELECT_SCENES) > MAX_NEW_IMAGES:
        raise ValueError("Job exceeds MAX_NEW_IMAGES. Narrow the selection before downloading models.")
    print("Persistent batch:", RUN_DIR, "New references:", new_ref_ids)
    ''')
    code('''
    # 5. Download/load ONLY after explicit opt-in. No automatic retry or second model download.
    if not RUN_GENERATION or not I_HAVE_READ_MODEL_LICENSES:
        raise RuntimeError("Enable RUN_GENERATION and read/acknowledge the model licenses in cell 3 first.")
    from diffusers import AutoPipelineForText2Image
    from huggingface_hub import HfApi
    api = HfApi()
    MODEL_REVISION = api.model_info(MODEL_ID).sha
    ADAPTER_ID = "h94/IP-Adapter"
    ADAPTER_REVISION = api.model_info(ADAPTER_ID).sha if USE_REFERENCE_CONDITIONING else None
    manifest.update(model_revision=MODEL_REVISION, adapter_revision=ADAPTER_REVISION,
                    adapter_id=ADAPTER_ID if USE_REFERENCE_CONDITIONING else None, adapter_license="Apache-2.0")
    save_manifest()
    CACHE_DIR = "/content/story_model_cache"  # Runtime cache; only images/manifests persist on Drive.
    load_start = time.perf_counter()
    kwargs = dict(torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True,
                  revision=MODEL_REVISION, cache_dir=CACHE_DIR)
    if FAMILY == "sdxl":
        kwargs["variant"] = "fp16"
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **kwargs)
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        def set_offload():
            if GPU_GIB < 8:
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
        set_offload()
        manifest["model_load_seconds"] = round(time.perf_counter() - load_start, 3)
        manifest["scheduler"] = {"class": type(pipe.scheduler).__name__, "config": dict(pipe.scheduler.config)}
        manifest["status"] = "model_loaded"
        save_manifest()
    except Exception as exc:
        manifest["errors"].append({"stage": "load", "error": str(exc)})
        save_manifest()
        raise RuntimeError("Model load failed. Stop; inspect the error. For memory failure restart the runtime and choose sd15. No automatic fallback download.") from exc
    ''')
    code('''
    # 6. Generation helpers. Save each successful image before continuing.
    adapter_loaded = False
    def ensure_adapter():
        global adapter_loaded
        if adapter_loaded or not USE_REFERENCE_CONDITIONING:
            return
        pipe.remove_all_hooks()
        weight = "ip-adapter_sdxl_vit-h.safetensors" if FAMILY == "sdxl" else "ip-adapter_sd15.safetensors"
        subfolder = "sdxl_models" if FAMILY == "sdxl" else "models"
        pipe.load_ip_adapter(ADAPTER_ID, subfolder=subfolder, weight_name=weight,
            image_encoder_folder="models/image_encoder", revision=ADAPTER_REVISION,
            cache_dir=CACHE_DIR, low_cpu_mem_usage=True)
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        set_offload()  # Enable offload after attaching the adapter/vision encoder.
        adapter_loaded = True
        manifest["adapter_weight"] = f"{subfolder}/{weight}"
        manifest["adapter_image_encoder"] = "models/image_encoder (ViT-H)"
        save_manifest()

    def board_for(identities):
        # One single-adapter image, not an incorrectly shaped list of adapters.
        board = Image.new("RGB", (512, 512), "#f4f4ee")
        width = 512 // len(identities)
        for n, identity in enumerate(identities):
            fitted = ImageOps.contain(refs[identity], (width - 12, 500))
            board.paste(fitted, (n * width + (width - fitted.width) // 2, (512 - fitted.height) // 2))
        return board

    def generate(kind, identity, prompt, seed, size, identities=()):
        existing = next((a for a in manifest["assets"] if a["kind"] == kind and a["target_id"] == identity), None)
        if existing:
            return Image.open(RUN_DIR / existing["file"]).convert("RGB"), existing
        if len(manifest["assets"]) >= MAX_NEW_IMAGES:
            raise RuntimeError("New-image budget exhausted")
        options = {}
        conditioning = "text_only"
        if kind == "scene" and identities and USE_REFERENCE_CONDITIONING:
            ensure_adapter()
            options["ip_adapter_image"] = board_for(identities)
            conditioning = "ip_adapter_contact_sheet"
        elif adapter_loaded:
            raise RuntimeError("Generate/reuse references before attaching the scene adapter; start a new batch for new references.")
        width, height = size
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                result = pipe(prompt=prompt, negative_prompt=request["negative_prompt"],
                    generator=torch.Generator(device="cpu").manual_seed(seed), width=width, height=height,
                    num_inference_steps=STEPS, guidance_scale=6.5 if FAMILY == "sdxl" else 7.5,
                    num_images_per_prompt=1, **options)
            if getattr(result, "nsfw_content_detected", None) and any(result.nsfw_content_detected):
                raise RuntimeError("Pipeline flagged this candidate. Review prompt; it will not be exported.")
            image = result.images[0].convert("RGB")
            relative = f"images/{kind}_{identity}_{seed}_{uuid.uuid4().hex[:8]}.png"
            path = RUN_DIR / relative
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                image.save(stream, format="PNG")
            metadata = {"prompt": prompt, "negative_prompt": request["negative_prompt"], "seed": seed,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_license": MODEL_LICENSE,
                "resolution": list(size), "runtime_seconds": round(time.perf_counter() - start, 3),
                "inference_steps": STEPS, "guidance_scale": 6.5 if FAMILY == "sdxl" else 7.5,
                "conditioning": conditioning, "references": {c: ref_sources[c] for c in identities},
                "adapter_id": ADAPTER_ID if conditioning != "text_only" else None,
                "adapter_revision": ADAPTER_REVISION if conditioning != "text_only" else None,
                "adapter_weight": manifest.get("adapter_weight") if conditioning != "text_only" else None,
                "adapter_scale": IP_ADAPTER_SCALE if conditioning != "text_only" else None,
                "peak_cuda_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
            record = {"kind": kind, "target_id": identity, "file": relative,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "metadata": metadata}
            manifest["assets"].append(record)
            manifest["status"] = "partial"
            save_manifest()
            return image, record
        except Exception as exc:
            manifest["errors"].append({"kind": kind, "target_id": identity, "seed": seed, "error": str(exc)})
            save_manifest()
            gc.collect()
            torch.cuda.empty_cache()
            raise  # No costly retries or silent model changes.
    ''')
    code('''
    # 7. Generate only missing references, then inspect the previews.
    for n, identity in enumerate(chars_by_id):
        if identity in new_ref_ids:
            image, record = generate("character", identity, chars_by_id[identity]["prompt"], BASE_SEED + n, REF_SIZE)
            refs[identity] = image
            ref_sources[identity] = {"file": record["file"], "sha256": record["sha256"], "approval": "candidate"}
        if identity in refs:
            print(identity, chars_by_id[identity]["appearance"])
            display(ImageOps.contain(refs[identity], (512, 512)))
    print("Check Rabbit: white fur, pink inner ears, BLUE VEST. Tortoise: green skin, BROWN patterned shell.")
    ''')
    md('''
    ## Inspect before spending another image

    If either reference is wrong, stop here, export the partial batch if useful, and disconnect the GPU.
    Change the character prompt/Bible for the next attempt rather than generating all scenes.
    For acceptable references, set `REFERENCES_OK=True` in the next cell and run it once.
    This preview acknowledgement is not local asset approval; imported files remain candidates until you approve their exact IDs.
    ''')
    code('''
    # 8. First quality test: exactly one scene. Edit True only after reviewing cell 7.
    REFERENCES_OK = False
    if not REFERENCES_OK:
        raise RuntimeError("Review reference previews, then set REFERENCES_OK=True in this cell.")
    for identity in SELECT_SCENES:
        scene = scenes_by_id[identity]
        if USE_REFERENCE_CONDITIONING and set(scene["characters"]) - set(refs):
            raise ValueError("Missing character references")
        # Scene seed depends on its persistent id, not selection/order; independently reproducible.
        seed = (BASE_SEED + int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)) % 2**32
        image, record = generate("scene", identity, scene["prompt"], seed, SCENE_SIZE,
                                 tuple(scene["characters"]) if USE_REFERENCE_CONDITIONING else ())
        print(identity, record["metadata"])
        display(ImageOps.contain(image, (1024, 576)))
    manifest["status"] = "complete"
    save_manifest()
    ''')
    code('''
    # 9. Export the current batch (also works for a partial reference-only batch).
    archive_base = Path("/content") / RUN_DIR.name
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=RUN_DIR)
    persistent_zip = RUN_ROOT / (RUN_DIR.name + ".zip")
    with open(archive, "rb") as source, persistent_zip.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    print("Saved permanently:", RUN_DIR, "and", persistent_zip)
    files.download(archive)
    print("Disconnect/delete the GPU runtime now to stop idle credit use.")
    ''')
    md('''
    ## Import, approve, and render locally

    Extract the downloaded ZIP into `imports/first_quality`. It contains `manifest.json`, the generated images,
    the exact request, per-image prompts/seeds/settings, model commit IDs, runtime and dependency versions.
    Run `python -m src.assets import-batch imports/first_quality/manifest.json`, then
    `python -m src.assets review`. Open the contact sheet and review JSON; approve only satisfactory asset IDs.
    No approval is inferred from filenames, generation order or notebook previews.

    For **one scene regeneration**, approve the character references locally first. If the scene is locked,
    explicitly unlock it with a reason. Export `python -m src.assets export-request --scene-ids finish`.
    Start a fresh notebook session, select `SELECT_SCENES=["finish"]`, `GENERATE_REFERENCES=False`,
    `MAX_NEW_IMAGES=1`; the request supplies the approved references. Generate one new candidate in a new batch.
    Existing reference or scene files are never replaced. Generation with the same seed may vary across hardware/library versions.

    For the full planned batch, select four scene IDs and set `MAX_NEW_IMAGES=6` (two refs + four scenes),
    or `4` with approved refs and `GENERATE_REFERENCES=False`. Do this only after the first quality test.
    If SDXL runs out of memory: export any partial results, disconnect/delete runtime, reconnect,
    choose `MODEL_FAMILY="sd15"`, keep 512px references and low-resolution scenes. Do not retry repeatedly.
    Adapter loading failures stop visibly. Set `USE_REFERENCE_CONDITIONING=False` only for an explicit text-only
    comparison (log marks it text-only); no claim of reference-conditioned consistency applies then.
    ''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}, "colab": {"name": "phase2_image_generation.ipynb"},
                 "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    for n, cell in enumerate(cells):
        cell["id"] = f"phase2-{n:02d}"
    destination = Path(__file__).resolve().parents[1] / "notebooks/phase2_image_generation.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    build()
