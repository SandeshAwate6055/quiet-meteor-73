# Animation & Motion Architecture Plan

## What Is Currently Implemented
The current local rendering pipeline (`src/media.py`, `src/production_providers.py`, `src/production.py`, `src/shorts.py`) implements **camera-only motion** via FFmpeg `zoompan`:
- Supported presets (`src/schema.py` `MOTIONS`): `push_in`, `pull_out`, `pan_left`, `pan_right`, `hold`.
- Every scene image is a static 16:9 illustration (`1024×576`) rendered with subtle camera movement, 0.18s fade-in/fade-out transitions, and narration audio synchronization (`max(scene.duration, wav_duration + tail_seconds + pause_after)` rounded up to `24 fps`).
- **No true character animation** (limb/facial/lip movement) is rendered locally by this pipeline.

## Hybrid Animation Architecture (Technical Implementation Plan)

### 1. Shot Mapping: 8 Character-Motion Candidates vs. 21 Camera-Only Shots
In `scripts/episode_01_golden_acorn.json` (29 shots total, ~220.33s measured SAPI runtime):
- **21 shots (`animation_intent: "camera_only"`)**: Rendered locally via `FFmpegAnimation` (`zoompan` presets) at **0 GPU cost**.
- **8 shots (`animation_intent: "character_motion_candidate"`)**: Action-heavy shots where an approved Image-to-Video (I2V) clip can be substituted:

| # | Shot ID | Characters | Source Image (`assets/library/...`) | Intended Character Motion | Target Clip Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `hook_blur` | `hare` | `candidates/scene_hook_blur_0807e753...png` | Hare sprints down sunny path, ears streaming, petals swirling | ~7.6s (loop/trim 5s I2V) |
| 2 | `crossroads_greeting` | `hare`, `tortoise` | `candidates/scene_crossroads_greeting_0aa7c8db...png` | Hare skids to a stop, Tortoise smiles & nods warmly | ~9.5s (5s I2V + hold/slow) |
| 3 | `go_signal` | `hare`, `tortoise` | `candidates/scene_go_signal_b9acaa72...png` | Hare springs forward in dust burst, Tortoise steps forward | ~4.3s (single 5s I2V clip) |
| 4 | `stone_slip` | `tortoise` | `candidates/scene_stone_slip_55a41112...png` | Tortoise wobbles on mossy stone, water splashes around feet | ~8.3s (5s I2V + slow settle) |
| 5 | `hare_sprints` | `hare` | `candidates/scene_hare_sprints_7d45ca6d...png` | Hare dashes frantically down trail, ears flying back | ~5.7s (5s I2V clip) |
| 6 | `hare_helps` | `hare`, `tortoise` | `candidates/scene_hare_helps_80002925...png` | Hare pushes flat stone & reaches paw out, Tortoise looks up | ~8.6s (5s I2V + hold) |
| 7 | `crossing_together` | `hare`, `tortoise` | `candidates/scene_crossing_together_30b47a1f...png` | Tortoise steps across stones while Hare steadies her | ~9.5s (5s I2V + slow pan) |
| 8 | `crossing_finish` | `hare`, `tortoise` | `candidates/scene_crossing_finish_aa87c58a...png` *(pending fix)* | Tortoise stretches across finish line, Hare cheers | ~8.4s (5s I2V + celebration) |

### 2. Recommended Open-Weight I2V Model & Licensing
- **Primary Candidate:** `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` (or `Wan-AI/Wan2.2-TI2V-5B-Diffusers` where supported in `diffusers`)
  - **License:** **Apache-2.0** (commercial use permitted, ungated weights, no paid API required).
  - **Resolution & FPS:** 480p (832×480) 16:9 native → upscaled cleanly via Lanczos to 1280×720 (`24 fps`), preserving character identity from the input keyframe (`FLUX.2-klein-4B` 1024×576 image).
- **Alternative Lightweight Candidate:** `Lightricks/LTX-Video` (open weights; check exact license terms before commercial deployment).

### 3. Colab GPU Usage & Cost Estimate (8 Clips)
- **Recommended Runtime:** Google Colab **A100 (40 GiB)** with High-RAM (or **L4 24 GiB** with CPU offload / FP8 quantization).
- **Estimated Time per 5s (81-frame) Clip:**
  - On **A100 40 GiB**: ~2.0–3.5 minutes per clip → **~16–28 minutes total** for all 8 action shots.
  - On **L4 24 GiB** (with CPU offload): ~4–6 minutes per clip → **~32–48 minutes total**.
- **Estimated Colab Cost:** ~4–12 Colab compute units for the 8-shot hybrid batch (vs. ~40+ units if animating all 29 shots).

### 4. Integration Design (`HybridAnimationProvider`)
1. **Colab I2V Worker (`src/colab_video.py`, future):**
   - Takes a SHA256-verified handoff ZIP containing only the 8 approved keyframe PNGs, prompts, seeds, and `generation_profile.json`.
   - Generates 1 MP4 clip per shot (`832×480`, 16–24 fps), writes `manifest.json` with SHA256 checksums + provenance, and exports a result ZIP.
2. **Local Import & Approval (`src/asset_store.py`):**
   - Extend `AssetStore` (`kind="clip"`) so generated I2V MP4s are imported as **unapproved candidates** and require human review + explicit approval before rendering.
3. **Hybrid Provider (`HybridAnimation` in `src/production_providers.py`):**
   - Implements the existing `AnimationProvider` protocol (`animate(image, audio, destination, duration, size, settings, motion)`).
   - If `store.approved("clip", scene.id)` exists and passes SHA256 verification: retimes/scales the approved I2V MP4 to `size` (`1280×720` or `720×1280` with blurred-background letterbox/fit for Shorts) and muxes the measured `audio` WAV (`-t duration`, 0.18s fades) via FFmpeg.
   - Otherwise: falls back deterministically to `FFmpegAnimation` (`zoompan` camera motion on the approved still illustration).
   - `src.production`'s content-addressed cache key (`clip:{scene.id}:{format}`) automatically invalidates and rebuilds only the shots whose approved clip changes, reusing all narration and unaffected clips.
4. **Quality Gate:** True character animation must **not** be claimed until actual I2V clips are generated on Colab, visually inspected for character/anatomy consistency, and explicitly approved.
