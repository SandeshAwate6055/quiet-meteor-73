# Phase 2 Colab quality test

## Current quality gate: v4 one-click remaining four scenes (`one_click_remaining_scenes.ipynb`)

The v4 character references (`hare` and `tortoise`) and the first scene (`challenge`) passed visual review and are pinned as **approved** in `assets/library/catalog.json` (`character_hare_ecaba14795945516dbb9d5dcc2675cc1696374322dee53d9f9ea535efb1d9ec8`, `character_tortoise_613f0557a5efa72b3f0ddff9e5ae8314539e8a31eecae9a15b700ed8a675e3bb`, and `scene_challenge_899664d4716f0f3272bb0247c3fc1e8727d559b536fd6edb3f94fa69e850840f`).

Use **[`notebooks/one_click_remaining_scenes.ipynb`](one_click_remaining_scenes.ipynb)** ([Open in Google Colab](https://colab.research.google.com/github/SandeshAwate6055/quiet-meteor-73/blob/master/notebooks/one_click_remaining_scenes.ipynb)):
1. Select **Runtime > Change runtime type > L4 or A100 GPU** (with **High-RAM**).
2. Click **Runtime > Run all**.
3. Wait for the notebook to reuse the 2 approved character references, preserve `challenge`, generate the **4 remaining scenes** (`racing_ahead`, `nap`, `steady_steps`, `finish` at `1024×576`), display the contact sheet and scene candidates inline, and trigger the result ZIP download automatically.

## Completed quality gate: v4 one-click `challenge` scene (`one_click_challenge_scene.ipynb`)

The v4 character references (`hare` and `tortoise`) passed visual review, were imported into `assets/library`, and are pinned as **approved** (`character_hare_ecaba14795945516dbb9d5dcc2675cc1696374322dee53d9f9ea535efb1d9ec8` and `character_tortoise_613f0557a5efa72b3f0ddff9e5ae8314539e8a31eecae9a15b700ed8a675e3bb`).

Use **[`notebooks/one_click_challenge_scene.ipynb`](one_click_challenge_scene.ipynb)** ([Open in Google Colab](https://colab.research.google.com/github/SandeshAwate6055/quiet-meteor-73/blob/master/notebooks/one_click_challenge_scene.ipynb)):
1. Select **Runtime > Change runtime type > L4 or A100 GPU** (with **High-RAM**).
2. Click **Runtime > Run all**.
3. Wait for the notebook to reuse the 2 approved character references, generate **only** the `challenge` scene (`1024×576`), display the scene and contact sheet inline, and trigger the result ZIP download automatically.

No separate CPU dry run, no variable edits, no manual ZIP upload, and no Google Drive login are required.

## Completed quality gate: v4 one-click character references (`one_click_generation.ipynb`)

Both v2 and v3 SDXL character references failed visual review (v3 rabbit had a pink bow tie, blue shorts, and frontal pose; v3 tortoise had grayish skin, an olive shell with abnormal geometry, and mismatched realism). All historical v2/v3 artifacts remain preserved and unapproved.

Use **[`notebooks/one_click_generation.ipynb`](one_click_generation.ipynb)** ([Open in Google Colab](https://colab.research.google.com/github/SandeshAwate6055/quiet-meteor-73/blob/master/notebooks/one_click_generation.ipynb)):
1. Select **Runtime > Change runtime type > L4 or A100 GPU** (with **High-RAM**).
2. Click **Runtime > Run all**.
3. Wait for the 2 character reference images (`hare` and `tortoise`) and contact sheet to display and the result ZIP to download automatically.

No separate CPU dry run, no variable edits, no manual ZIP upload, and no Google Drive authentication are required. Temporary Colab storage (`/content/story_one_click`) is used by default and lost on disconnect; optional persistent storage (`USE_GOOGLE_DRIVE = True`) is available in Cell 1. The v4 profile ([`configs/one_click_generation.json`](../configs/one_click_generation.json)) uses [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`, **Apache-2.0** license, 4 steps, 1024×1024, BF16) while preserving `ColabDiffusersBackend` (SDXL/SD1.5) in the shared worker. **Zero scenes are generated; generated images are unapproved candidates until human review.**

## Historical quality gate: v3 references only

The real v2 references failed visual review. Use **`examples/character_references_v3.zip`** with a fresh copy of `phase2_batch_worker.ipynb`. [Exact v3 settings and retry instructions](REFERENCE_V3.md) cover CPU dry run, two text-only SDXL character images, review/export and stop. **Zero scenes are requested; challenge, smoke and full modes are blocked.** No image is automatically approved. V3 has not run on GPU; local synthetic tests do not establish visual quality.

The actual v2 manifest shows text-only reference generation, no active adapter, and untruncated prompts. The saved contact sheet was inspected. [The diagnosis](REFERENCE_V3.md#what-the-saved-run-establishes) distinguishes these facts from prompt-pattern/color-binding hypotheses. The old v1/v2 packages and images remain historical and unchanged. The active notebook rejects them; older/general workflow details below are retained for context and are not current generation instructions.

## Historical/general worker: all requested scenes, with smoke-first resume

The original batch workflow used [phase2_batch_worker.ipynb](phase2_batch_worker.ipynb), which is now updated for the restricted v2 smoke above. Two real historical references remain unapproved. A later real challenge smoke was reported technically successful but visually unacceptable. The v2 worker generated two references on Colab, both rejected visually; no v2 challenge was generated. The **v3 reference-only revision has not been GPU-executed**. Shared state/intake and backend argument wiring are tested locally with synthetic fixtures; no model, ML package or GPU credit was used during this revision.

Export a ready handoff without modifying existing production or rendering:

```powershell
.\.venv\Scripts\python.exe -m src.production --output-dir outputs/production_demo --prepare-image-batch --handoff-dir outputs/batch_generation_handoff
```

Open Colab → File → Upload notebook → choose `phase2_batch_worker.ipynb` → save a copy to Drive. Begin with a CPU runtime. Run cells 1–3 using `MODE="dry_run"` and upload the printed handoff ZIP. It contains the full five-scene request, available references with original metadata, checksum index and exactly the shared worker source that was locally tested. No repository clone is needed. It persists under `MyDrive/AI_Kids_Story_Studio/batch_handoff/`; save the printed `PACKAGE_PATH` for subsequent sessions.

Preflight shows existing images to reuse and images to create. Current references are candidates; by default this blocks generation instead of replacing them. Review the actual references first: Rabbit has two views and a teal-blue vest, while the front-facing Tortoise reference does not establish its brown back shell. Approve only if satisfactory, then rerun production/export so approved references are embedded. Alternatively set `ALLOW_UNAPPROVED_REFERENCE_PREVIEW=True` explicitly for experimental preview; the worker warns and tags resulting scenes. This neither approves references nor makes generated scenes eligible for production automatically.

For the first GPU smoke:

```python
MODE = "smoke"
MAX_NEW_IMAGES = 1  # both existing references reused
RUN_GENERATION = True
INSTALL_COLAB_DEPS = True
LICENSES_REVIEWED = True
ALLOW_UNAPPROVED_REFERENCE_PREVIEW = True  # only if deliberately previewing current candidates
QUALITY_REVIEWED = False
```

Select Runtime → Change runtime type → GPU manually, keep the saved `PACKAGE_PATH`, and run cells 1–5. With absent references, preflight instead requires a budget of 3 and creates only those missing references before the smoke scene. Existing reference files and successful images are never overwritten.

After inspecting the smoke contact sheet, keep the same package/result folder and set:

```python
MODE = "full"
QUALITY_REVIEWED = True
MAX_NEW_IMAGES = 4  # four remaining scenes; successful smoke/reference images do not count
```

Rerun cells 1–5. The worker processes ALL scene IDs from the request (`challenge`, `racing_ahead`, `nap`, `steady_steps`, `finish` for this production), preserves deterministic per-ID seeds, logs failures including adapter setup, and saves each success before proceeding. A failed invocation stops at its first error, preserving completed jobs. Retry only after inspecting the error and budget. `FAMILY="sd15"` is the smaller explicit fallback, with its matching adapter and 512px reference / 768×432 scene sizes; switching model/settings requires a NEW `RESULT_BATCH_ID`. SDXL uses 768px refs / 1024×576 scenes, fp16, VAE tiling/slicing and CPU offload. No conditioning-disable option is provided for scenes with characters.

An interruption resumes from the same Drive directory/request/settings; the model revision from the first successful load is reused. A hard runtime kill may leave `.worker.lock`: verify no worker session is active before manually removing that stale lock. Normal exceptions and keyboard interruption release the lock and persist errors. No automatic OOM retries or second-model downloads occur.

Cell 5 creates one labeled sheet covering references and every scene, including missing/failed panels. Cell 6 exports a new uniquely named ZIP, including `request.json`, compatible version-1 `manifest.json`, images, copied inputs, review sheets and state history. It also works for partial/failed batches. Completion is recomputed from actual checksum-verified scene files; a smoke run with four remaining scenes is `awaiting_smoke_review`, never `complete`. Disconnect/delete the GPU runtime after finishing to stop idle credit use.

Import the ZIP using the existing production command; no manual extraction is required:

```powershell
$batchResult = Read-Host 'Full path to downloaded result ZIP'
.\.venv\Scripts\python.exe -m src.production --output-dir outputs/production_demo --import-image-batch $batchResult
```

The printed receipt gives imported IDs, one or more contact sheets, review JSON, missing scenes and failed scenes. Exact existing approval workflow:

```powershell
$acceptedAssetId = Read-Host 'Exact accepted asset_id from review.json'
.\.venv\Scripts\python.exe -m src.assets approve $acceptedAssetId
.\.venv\Scripts\python.exe -m src.production --script scripts/tortoise_and_hare.json --output-dir outputs/production_demo
```

Repeat approval for every accepted image. Leave rejected images unapproved; use existing `unlock` with a reason to replace an already pinned target. Production continues to select only approved scene assets and to preserve prior clips/exports. Request binding accepts results from current or retained historical requests for that production; importing another story's result is rejected.

The worker source is `src/image_batch.py`; the Colab-only adapter is `src/colab_images.py`, adapted from the original notebook's model/adapter workflow. Source packaging and result intake are in `src/batch_handoff.py` and reuse `src.assets` and `AssetStore`. Code/checkpoints/pins are not downloaded onto Windows. [Diffusers 0.35.1 adapter documentation](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/ip_adapter) documents the architecture-specific SDXL and SD 1.5 pairings; published model/license links remain below. GPU memory fit, image quality and identity consistency still require actual Colab execution and review.

## Historical single-scene template

The saved `phase2_image_generation.ipynb` template remains unexecuted in the repository. The actual earlier exported batch contains two real references, not a generated scene. The old workflow's `REFERENCES_OK=False` reset and adapter setup outside the error handler can explain a stop with no saved scene/error, but the exact historical stop is not recoverable without its execution log. Its setup also creates a new batch directory rather than resuming one. The new worker addresses these code defects; no claim is made that any one was proven to cause that earlier run.

## First small batch: two references + one scene

1. From `D:\AI_Kids_Story_Studio`, export the Character Bible and four scene prompts:

   ```powershell
   .\.venv\Scripts\python.exe -m src.assets export-request
   ```

   This creates `notebooks/colab_input.json`. It includes Rabbit under persistent ID `hare`, Tortoise under `tortoise`, style/environment, deterministic prompt inputs, and any approved reference images embedded in the request. The sample scene IDs remain unchanged.

2. Open [Google Colab](https://colab.research.google.com/). Choose **File > Upload notebook**, select `notebooks/phase2_image_generation.ipynb`, then **File > Save a copy in Drive**.

3. Choose **Runtime > Change runtime type**, select **Python 3** and a **GPU** (T4 is a reasonable first attempt). Connect. Run cells individually; avoid **Run all**. GPU runtime time can consume credits even while idle.

4. In code cell **1**, set `INSTALL_COLAB_DEPS=True`, then run it. Packages install only in Colab. Keep its preinstalled CUDA PyTorch. Restart the session if Colab requests it, then proceed to cell **2** to inspect GPU and available RAM.

5. Read the model-license links in the opening markdown. In cell **3**, set:

   ```python
   RUN_GENERATION = True
   I_HAVE_READ_MODEL_LICENSES = True
   MODEL_FAMILY = "auto"
   SELECT_SCENES = ["challenge"]
   GENERATE_REFERENCES = True
   MAX_NEW_IMAGES = 3
   ```

   Leave the remaining settings unchanged. `auto` chooses SDXL only when the notebook sees at least 14 GiB GPU memory and 10 GiB free host RAM. These thresholds are estimates, not a guarantee that a particular runtime will fit. The smaller fallback is SD 1.5 with its matching adapter, not SDXL weights loaded into an incompatible architecture.

6. Run cell **4**. Mount your Drive and upload the exported `colab_input.json`. Each batch gets a new unique folder under `MyDrive/AI_Kids_Story_Studio/image_batches/`. Model caches stay in the disposable Colab runtime; images, exact request and manifests persist on Drive.

7. Run cells **5–7**. These download/load the selected checkpoint, define generation helpers, and generate **only two references**. Inspect the previews: Rabbit must have white fur, pink inner ears and a blue vest; Tortoise must have green skin and a brown patterned shell. If either is wrong, stop before scenes. Cell **9** can export this partial batch. Disconnect/delete the runtime when finished.

8. If references are acceptable, set `REFERENCES_OK=True` in cell **8** and run it once. This generates **only the challenge scene**. The IP-Adapter receives a combined reference board; it may blend attributes or miss a character. Check identity, colors, clothing, anatomy, friendly mood and composition. Consistency is not guaranteed.

9. Run cell **9** to save/download the ZIP. Then choose **Runtime > Disconnect and delete runtime** to stop idle GPU use. A copy of the ZIP and its unpacked batch remain in Drive.

No automatic model substitution or OOM retry is performed. If SDXL fails for memory, export any partial batch, delete/reconnect the runtime, set `MODEL_FAMILY="sd15"`, and repeat the small test. SD 1.5 uses 512px references and 768×432 scenes (512×384 on GPUs below 8 GiB), fp16, VAE tiling/slicing and CPU offload. Quality and prompt adherence can be lower. Inspect exception details first; network or access errors are not evidence of a memory problem.

## Import and approve locally

Extract the ZIP into `imports/first_quality/` so that `manifest.json` sits directly in that folder. Then run:

```powershell
.\.venv\Scripts\python.exe -m src.assets import-batch imports/first_quality/manifest.json
.\.venv\Scripts\python.exe -m src.assets review
```

Open `outputs/asset_review/contact_sheet_001.jpg` and `outputs/asset_review/review.json`. The JSON includes full candidate IDs and original generation metadata. Inspect the original full-resolution images too: `assets/library/` stores each imported candidate at the relative path given by its catalog entry. The contact sheet lists the same candidates by review number and checksum prefix.

Approve satisfactory references and scenes individually using the **exact IDs** from the review JSON:

```powershell
.\.venv\Scripts\python.exe -m src.assets approve character_hare_REPLACE_WITH_FULL_HASH
.\.venv\Scripts\python.exe -m src.assets approve character_tortoise_REPLACE_WITH_FULL_HASH
.\.venv\Scripts\python.exe -m src.assets approve scene_challenge_REPLACE_WITH_FULL_HASH
```

The suffixes above are placeholders; paste the full IDs returned by import/review. Imports never approve or replace a locked selection. Files are created exclusively, checksummed, and the approved choice remains pinned even if newer candidates arrive. Locks are enforced by this application, not OS access permissions; manual file changes are detected and cause rendering fallback. Back up the whole `assets/library/` folder, including `catalog.json`, plus the Character Bible.

Render with approved illustrations where available and Pillow for the other scenes:

```powershell
.\.venv\Scripts\python.exe -m src.main --script scripts/tortoise_and_hare.json --output-dir outputs/phase2_first_quality
```

The renderer contains the entire illustration in both orientations, adds readable narration captions, and pads with a blurred background. This protects character framing; it does not invent missing vertical artwork or new detail. Small images are upscaled with Lanczos, not AI. The output manifest records each selected asset ID/checksum, source dimensions and fallback reason. Phase 1 outputs stay at their original paths.

## Regenerate one scene independently

Approve reference images first. If the scene is locked, unlock its selection explicitly; this retains every image and adds an audit entry:

```powershell
.\.venv\Scripts\python.exe -m src.assets unlock scene challenge --reason "Rabbit vest needs correction"
.\.venv\Scripts\python.exe -m src.assets export-request --scene-ids challenge
```

Start a new notebook batch with the fresh request, `SELECT_SCENES=["challenge"]`, `GENERATE_REFERENCES=False`, and `MAX_NEW_IMAGES=1`. Approved references arrive embedded in the request. If you leave a scene locked, the notebook refuses to generate it. Nothing overwrites its approved file. To compare a replacement while retaining the old approval, generate candidates before initial approval, or retain the old asset ID and reapprove it after review. Import the new batch, review, and approve the chosen candidate explicitly.

For the planned four scene illustrations, use `SELECT_SCENES=["challenge", "racing_ahead", "steady_steps", "finish"]`. Set `MAX_NEW_IMAGES=4` with approved references and `GENERATE_REFERENCES=False`, or `6` for two new refs plus four scenes. Unlock any approved selected scene before exporting a request. The fifth scene (`nap`) stays as a Pillow card; request it separately if desired. Never expand the batch until the first quality test looks acceptable.

## Checkpoints, licenses and compatibility

The following were checked against publisher/model repository information during implementation:

| Component | Checkpoint | Published license | Conditioning |
| --- | --- | --- | --- |
| Preferred | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md) | SDXL `ip-adapter_sdxl_vit-h.safetensors` |
| Smaller fallback | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | CreativeML OpenRAIL-M | SD 1.5 `ip-adapter_sd15.safetensors` |
| Adapter + encoder | [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) | Apache-2.0 | `models/image_encoder` ViT-H; no FaceID |

Use restrictions apply to the base checkpoints; open weights do not mean unrestricted licensing. The SD 1.5 repository identifies itself as a community mirror. The notebook resolves and records exact model/adapter commit IDs at runtime. It retains the SD 1.5 pipeline's supplied safety checker. SDXL Base has no equivalent built-in checker here; human review is required before approving children's content.

The workflow follows the [versioned Diffusers adapter API](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/ip_adapter), the [loader implementation](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/loaders/ip_adapter.py), and [memory documentation](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory). Pinned Colab dependencies and published compatibility are preparation evidence; actual model loading, GPU memory fit, runtime, generated quality and consistency remain unverified until you execute it. Seeds aid reproducibility but cannot guarantee identical output across hardware or dependency changes.

`scripts/build_colab_notebook.py` rebuilds the notebook JSON without executing any cells. Notebook syntax/default-budget tests run with the normal local suite. It is not a substitute for a real Colab run.
