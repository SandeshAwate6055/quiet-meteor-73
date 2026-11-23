# AI Kids Story Studio

A local-first Python project for turning structured children's stories into narrated landscape videos and vertical Shorts. The current proof of concept combines offline speech, Pillow/approved illustrations, FFmpeg, and a resumable production orchestrator. A manually started Colab worker supplies candidate AI illustrations; the longer-term goal is polished 3–5-minute animated stories.

## Public checkout and Colab inputs

[Open the batch notebook in Google Colab](https://colab.research.google.com/github/SandeshAwate6055/quiet-meteor-73/blob/master/notebooks/phase2_batch_worker.ipynb) · [Download the portable sample handoff](https://raw.githubusercontent.com/SandeshAwate6055/quiet-meteor-73/master/examples/colab_handoff.zip)

Install the local dependencies using the setup section below. The batch notebook is [notebooks/phase2_batch_worker.ipynb](notebooks/phase2_batch_worker.ipynb). A small, audited, portable [Colab handoff ZIP](examples/colab_handoff.zip) is included so a fresh checkout can perform its dry run without any ignored local outputs. See [package contents and smoke-test steps](examples/README.md).

The sample ZIP includes the five-scene request, the shared worker source, and two generated character references labeled **unapproved**. Local absolute paths were removed from its copied metadata; generation prompts, seeds, model details and image bytes/checksums were preserved. The original local ZIP remains unchanged. No videos, model weights, private asset catalog, credentials or machine-specific configuration are included. In Colab upload this sample ZIP when cell 2 prompts for a handoff. Default dry-run mode performs no model download or inference; GPU runtime start, preview consent and quality/asset approval remain manual.

## Batch AI images: production → Colab → review → production

1. Run production normally:

   ```powershell
   .\.venv\Scripts\python.exe -m src.production --script scripts/tortoise_and_hare.json --output-dir outputs/production_demo
   ```

   It now publishes an immutable `image_handoff/request_*.zip` containing the missing-scene request, shared worker code, and available character references. The manifest/console prints `image_batch_handoff.package_path`. To export from the existing production **without rendering or changing its state**, use:

   ```powershell
   .\.venv\Scripts\python.exe -m src.production --output-dir outputs/production_demo --prepare-image-batch --handoff-dir outputs/batch_generation_handoff
   ```

2. Upload [the batch worker notebook](notebooks/phase2_batch_worker.ipynb) to Colab. Run cells 1–3 on a CPU runtime with default `MODE="dry_run"`; upload the single handoff ZIP when prompted. It validates inputs/references and shows every requested scene, reused references, new-image count, budget and blockers before any model load. Save the printed `PACKAGE_PATH` for resume. The original smoke-test notebook is retained as a historical template.

3. Start a GPU runtime manually only when ready. For one-scene smoke generation set `MODE="smoke"`, `RUN_GENERATION=True`, `INSTALL_COLAB_DEPS=True`, `LICENSES_REVIEWED=True`, and `MAX_NEW_IMAGES=1` when reusing both existing references. Those references are unapproved: either approve acceptable references locally and export a fresh request, or explicitly set `ALLOW_UNAPPROVED_REFERENCE_PREVIEW=True` for experimental candidates. This does not approve any assets. Run cells 1–5, inspect the contact sheet, then use `MODE="full"`, `QUALITY_REVIEWED=True`, and `MAX_NEW_IMAGES=4` to generate the four remaining scenes. Keep the same package/result directory and rerun cells 1–5. If no references exist, the smoke budget needs 3 (two references plus one scene). The worker saves each image immediately, reuses verified successes, and never silently disables conditioning. Full mode cannot proceed without a successful reviewed smoke, and a five-scene request cannot be marked complete with missing/failed scenes.

4. Export with cell 6, disconnect/delete the GPU runtime, then import the downloaded ZIP directly:

   ```powershell
   $batchResult = Read-Host 'Full path to downloaded batch result ZIP'
   .\.venv\Scripts\python.exe -m src.production --output-dir outputs/production_demo --import-image-batch $batchResult
   ```

   Intake validates checksums, decoded images, IDs, prompts, seeds, conditioning, request binding and completeness before existing candidate import/review APIs. It prints contact-sheet paths and missing/failed scenes. Partial results remain partial; duplicate intake does not duplicate records. Successful results from older requests belonging to the same production remain importable after approving the smoke and replanning. Original files, selected approvals and earlier videos remain untouched.

5. Review the sheet and full-resolution candidates, then approve only accepted IDs and rerun production:

   ```powershell
   $acceptedAssetId = Read-Host 'Exact accepted asset_id from the printed review.json'
   .\.venv\Scripts\python.exe -m src.assets approve $acceptedAssetId
   .\.venv\Scripts\python.exe -m src.production --script scripts/tortoise_and_hare.json --output-dir outputs/production_demo
   ```

   Repeat the approval command for each accepted reference/scene. Approval locks and checksums are unchanged; rejected candidates stay unapproved. Newly approved scenes rebuild their affected images/clips/exports while unrelated stages and narration are reused. See [batch notebook details](notebooks/README.md).

The local state machine, result intake and renderer integration are tested with **synthetic fixtures**. The new GPU backend/notebook has **not been executed on Colab**. No runtime-control capability is implemented: connecting, starting and stopping Colab, experimental-reference consent, smoke quality review, downloading one ZIP, and local asset approval remain manual. SDXL/SD 1.5 generation uses the existing documented IP-Adapter pairing and licenses; consistency and memory fit require the actual smoke test. No animation models or Windows GPU dependencies were added.

## Master production orchestrator

Run a resumable production in a **new** directory:

```powershell
.\.venv\Scripts\python.exe -m src.production --script scripts/tortoise_and_hare.json --output-dir outputs/production_demo
```

Rerun the exact command to resume. `production_manifest.json` reports per-scene narration, images, clips, asset approval/fallback state, output paths, timings, errors, and executed/reused stages. Final MP4 paths are listed under `outputs.landscape.path` and `outputs.vertical.path`; the compatible Colab request is under `ai_generation_request.path`. No Colab session is launched and no assets are approved. The imported Rabbit/Tortoise references are still unapproved; no challenge scene exists, so the current sample uses Pillow cards for all five scenes.

Settings are optional: pass `--settings configs/production.json` (the CLI default). It supports advisory `target_duration` in seconds, English `language`, future-image-prompt `style`, `formats` (`landscape`, `vertical`), `vertical_scene_ids`, narration `rate`/`silent`, and the implemented `zoom_pan` animation. Target duration never stretches or invents scenes. Large differences between the target, narration and final duration produce warnings. This structured-script planner performs validation and scheduling, not intelligent story generation or translation. A style override changes future image prompts; existing illustrations/placeholders are not restyled. Other languages and animation types are rejected explicitly until their providers exist. FFmpeg quality remains the existing CRF 22 setting.

Successful stage artifacts are checked by SHA256 before reuse. Narration text/settings, resolved approved image hashes, caption/layout inputs, durations, render settings and provider/code revisions determine dependent work. Changed or corrupt inputs rebuild affected stages into **new attempt directories**; old media remains available. `runs/` preserves production snapshots, including failures. Voice availability/installed voices can change outside the project: use `--refresh-narration` to retry speech, including cached silent fallbacks. Ordinary reruns retain the refreshed voice selection. An interrupted process can leave `.production.lock`; verify no writer is active before manually removing that lock.

`src/production_plan.py` plans; `production_state.py` handles checkpoints/cache; `production_providers.py` defines planning, image resolution/request preparation, voice, animation and final-rendering contracts/adapters. `src/production.py` coordinates them. Existing `src.main`, Pillow/approved-file providers, narration, FFmpeg rendering, approval locks and checksum checks are reused. Final exports receive metadata validation and a full decode before completion; unchanged exports reuse that validation only after their hashes match. Missing/corrupt approved files fall back explicitly, and damaged approval pins remain locked in the prepared request until manually resolved. Unapproved references are never embedded as approved conditioning images. Review candidates before requesting replacements.

Automated now: structured planning, asset resolution, batch handoff preparation, sequential image generation/resume inside a manually started Colab worker, result validation/import/review, offline narration, zoom/pan rendering, assembly, validation and production resume. Colab runtime start/stop and asset approval remain manual. Image-to-video motion, lip sync, translation, semantic story planning and UI are future integrations. No new Windows dependencies are required.

Phase 2 adds a Character Bible, reviewable image imports, locked approvals, and an approved-file image provider. See [the Colab quality-test guide](notebooks/README.md) and [notebook](notebooks/phase2_image_generation.ipynb). A real partial Colab batch supplied two imported, unapproved character references; its challenge scene is missing. The Windows dependency list is unchanged.

The legacy `src.main` command writes new renders under `outputs/phase2/` by default, keeping the verified Phase 1 videos in `outputs/tiny_steps_big_finish/`. Existing narration, FFmpeg timing, transitions and validation are retained. Approved scene images are selected from `assets/library/`; missing, unapproved or modified images fall back to the original Pillow cards. Use `--image-provider pillow` with `src.main` to force placeholders, or `--asset-dir path/to/library` for a separate asset collection.

A local, model-free proof of concept for educational story videos. The original five-scene retelling **Tiny Steps, Big Finish** produced a 42.48-second landscape MP4 and a 26.35-second vertical highlight in the verified local run. It uses Pillow illustrations, installed Windows speech voices, and CPU-only FFmpeg. No credentials, APIs, model downloads, CUDA, or paid services are used.

## Windows setup (Python 3.10+)

From this folder in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

An existing working environment can be reused; the initial local run used Python 3.14.6. Pillow is the only required Python dependency. Windows SAPI needs no additional Python package. Optional secondary speech engine:

```powershell
.\.venv\Scripts\python.exe -m pip install pyttsx3
```

Install FFmpeg **and FFprobe** if missing:

```powershell
winget install --id Gyan.FFmpeg -e
```

Reopen PowerShell after installation. Both binaries must be on PATH, or pass `--ffmpeg C:/tools/ffmpeg/bin/ffmpeg.exe --ffprobe C:/tools/ffmpeg/bin/ffprobe.exe`. Alternatively set `FFMPEG_BINARY` / `FFPROBE_BINARY`, or put these two paths in ignored `configs/local.json`:

```json
{"ffmpeg": "C:/tools/ffmpeg/bin/ffmpeg.exe", "ffprobe": "C:/tools/ffmpeg/bin/ffprobe.exe"}
```

This workspace's local configuration points to the existing FFmpeg installation in the `mfa` micromamba environment. That environment is used only for its media executables; no models are loaded. Configuration precedence: CLI, environment variables, local paths, PATH.

## Run everything

```powershell
.\.venv\Scripts\python.exe -m src.main --script scripts/tortoise_and_hare.json
```

Or, after activating your environment:

```powershell
python -m src.main --script scripts/tortoise_and_hare.json
```

Phase 1 files remain in `outputs/tiny_steps_big_finish/`; new default renders live in `outputs/phase2/tiny_steps_big_finish/`:

- `landscape.mp4`: 1280×720, 24 fps, H.264 video and AAC narration.
- `vertical.mp4`: 720×1280 highlight using challenge, steady_steps, finish, in story order.
- `images/`: five landscape and three portrait Pillow scene cards.
- `audio/`: per-scene WAV narration and plain-text speech input.
- Per-scene MP4 clips, concat lists, and `manifest.json` with actual durations, speech backends, warnings, and validation results.

Rerunning the same title overwrites its generated files; use `--output-dir outputs/another_run` to keep a separate run. Rendering runs sequentially with two CPU threads and one filter thread; no GPU memory is needed.

`--vertical-scenes challenge finish` changes the highlight selection. A script with different scene IDs defaults to its last three scenes. `--silent` explicitly tests the fallback. `--config path/to/config.json` supplies pipeline settings.

## Project layout

```text
configs/       Pipeline defaults, JSON scene schema, ignored local tool paths
assets/        Approved/candidate library, optional reference paths, reusable art
scripts/       Structured story JSON
src/
  schema.py    Validated Story and Scene data classes
  characters.py  Character Bible parsing and reusable scene prompt building
  asset_store.py  Checksummed candidates, approval locks and audit history
  assets.py    Import, contact-sheet review, approval and Colab-request CLI
  image_providers.py  Pillow fallback and approved-file illustration providers
  visuals.py   Pillow cartoon drawing and responsive scene cards
  narration.py Windows SAPI -> optional pyttsx3 -> marked silent fallback
  sapi_narrate.ps1  Offline Windows COM speech-to-WAV worker
  speech_worker.py  Isolated, timeout-bounded pyttsx3 worker
  media.py     FFmpeg rendering, FFprobe inspection, output validation
  main.py      Command-line workflow and manifest
outputs/       Generated media (ignored by Git)
notebooks/     Colab image generation and exact first quality-test instructions
tests/         Original regressions plus Phase 2 asset/provider/notebook tests
```

Each scene has a unique safe `id`, `narration`, requested `duration` in seconds, `visual_description`, and optional `characters` list. See `configs/scene_schema.json`; runtime validation requires no JSON Schema package. Example:

```json
{
  "id": "challenge",
  "narration": "Tortoise invited Hare to a race.",
  "duration": 8,
  "visual_description": "A sunny meadow and a starting line.",
  "characters": ["tortoise", "hare"]
}
```

The whole document contains `title` and a `scenes` array. Keep narration short enough for readable captions. Requested durations are minimums: each scene extends to fit its speech plus 0.35 seconds, rounded up to a frame. This preserves narration; longer scripts or slower installed voices can exceed 45 seconds. Adjust narration, durations, or `speech_rate` (SAPI -10 to 10, default 0). The optional pyttsx3 fallback uses 150 words/minute.

Cards show narration throughout each scene, with subtle centered zoom and short fades through black. Portrait cards are redrawn to fit the screen. Placeholder visuals recognize the sample's scene IDs and tortoise/hare characters; `visual_description` is retained as a future provider prompt, not interpreted by an AI image generator.

If speech fails or times out, the pipeline logs each failure, substitutes a valid silent WAV, keeps narration visible on cards, and sets `spoken_narration: false` in the manifest. A silent audio track passes structural validation but does not count as successful spoken narration. No speech packages or voices are downloaded automatically.

## Tests and validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover malformed scripts, duplicate/unsafe identifiers, duration constraints, bounded configuration, speech failure, both image dimensions, and real FFmpeg rendering. Phase 2 adds Character Bible parsing, import/review/export, locked deterministic selection, missing/corrupt image fallback, resolution adaptation, mixed-provider video rendering, and notebook syntax/small-batch defaults. The production suite additionally exercises complete execution, unapproved-asset exclusion, failure/resume, idempotence, selective invalidation, damaged exports, advisory duration, and protection of existing output directories. Media integration tests skip if the executables are unavailable. Every new final export checks file size, positive duration, dimensions, audio coverage, timing, and full decoding. Legacy `manifest.json` and the new `production_manifest.json` retain per-scene image provenance.

## Not implemented yet

Full character animation, lip sync, AI video, music, word-timed subtitles, a user interface, and YouTube uploading are not implemented. Phase 2 adds the Colab image-generation notebook and local approved-image integration; two real reference images have been imported but remain unapproved, and scene generation is incomplete. This POC animates static illustrations or scene cards. Longer 3–5-minute production requires suitable narration and assets; the current sample remains approximately 42 seconds.
