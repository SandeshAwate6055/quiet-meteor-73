# Animation & Motion Architecture Plan

## What Is Currently Implemented
The current local rendering pipeline (`src/media.py`, `src/production_providers.py`, `src/production.py`) implements **camera-only motion** via FFmpeg `zoompan`:
- Supported presets (`src/schema.py` `MOTIONS`): `push_in`, `pull_out`, `pan_left`, `pan_right`, `hold`.
- Every scene image is a static 16:9 illustration (`1024×576`) rendered with subtle camera movement, fade-in/fade-out transitions, and narration audio synchronization.
- **No true character animation** (limb/facial/lip movement) is rendered locally by this pipeline.

## Hybrid Animation Plan (Next Milestone)
In `scripts/episode_01_golden_acorn.json` (29 shots total):
- **21 shots (`animation_intent: "camera_only"`)**: Establishment, dialogue reactions, resting, and emotional beats that work well with camera motion (`push_in`, `pull_out`, `pan_left`, `pan_right`, `hold`) at **0 GPU cost**.
- **8 shots (`animation_intent: "character_motion_candidate"`)**: Action-heavy moments (`hook_blur`, `crossroads_greeting`, `go_signal`, `stone_slip`, `hare_sprints`, `hare_helps`, `crossing_together`, `crossing_finish`).

### Practical Image-to-Video (I2V) Approach & Cost Estimate
- **Recommended Candidate Models (Colab GPU):** Open-weight I2V models such as `Wan-AI/Wan2.1-I2V-14B-480P` (Apache-2.0) or `zai-org/CogVideoX-5b-I2V` / `Lightricks/LTX-Video` (check exact license before use).
- **Compute & Cost Estimate (8 action shots × ~4s clips):**
  - On **Colab A100 (40 GiB)** or **L4 (24 GiB)** with 480p–576p I2V generation at 25–30 steps: ~1.5–4 minutes per clip = **~15–32 minutes of GPU time** (~3–10 Colab compute units) for all 8 action shots, vs. ~90+ minutes if animating all 29 shots.
  - Because `src/production.py` caches per-shot clips by content hash, any approved I2V clip can replace a `camera_only` clip without re-rendering narration or unaffected scenes.
- **Quality Gate:** Do not claim true character animation until actual I2V clips are generated, visually inspected for character consistency, and explicitly approved.
