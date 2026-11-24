# Character references v3: stop before scenes

Both v2 references failed human visual review. They remain historical candidates; no approval has changed. This revision prepares **only hare and tortoise references**, with zero scene jobs. It has **not run on a GPU**. Local tests use synthetic images and mocked inference, and exercise the existing renderer separately.

## What the saved run establishes

The local `reference_v2_failed/manifest.json` records SDXL Base 1.0, revision `462165984030d82259a11f4367a4eed129e94a7b`, NVIDIA L4, EulerDiscreteScheduler, 24 steps, guidance 6.5, 768-square images. Both records say `conditioning="text_only"`, with null adapter ID/weight/scale and empty reference maps. IP-Adapter/collage conditioning did **not** cause these particular reference failures.

Both tokenizers recorded 45 positive tokens for hare, 48 for tortoise, 37 negative tokens, against a limit of 77. Truncation is not supported as an explanation. The saved contact sheet was visually inspected: the rabbit has a pink vest/bow tie; the tortoises repeat across the image with green shells and a flatter illustration style. These agree with the user's rejection. No challenge was generated in that v2 run.

Plausible contributors, **not proven causes**:

- The old hare prompt specified blue but did not prohibit a pink vest or bow tie. Attribute binding/color spill and learned clothing associations may explain the result.
- The phrase `brown patterned shell` did not confine patterning to one shell. The model may have interpreted it as a repeating whole-image motif. Duplication negatives existed, but tiles/wallpaper/repeating motifs were not explicit.
- A generic cartoon suffix did not reliably establish a shared 3D studio-render treatment. Text-only SDXL is not an exact character/count/color constraint solver; its [model card documents compositional limitations](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0#limitations-and-bias).

No seed/prompt/resolution ablation has been run. V3 is a practical quality attempt, not evidence that any hypothesis is correct or that these problems are solved.

## V3 specification

Exact positive and negative strings are in [the request](../examples/character_references_v3.json) and the Character Bible. Scene prompt construction and the scene-conditioning backend are unchanged.

- Hare: exactly one white rabbit, pink inner ears, small pink nose, royal blue sleeveless vest as its only clothing, full-body three-quarter view.
- Tortoise: exactly one green tortoise, full-body three-quarter side view; one large rounded brown back shell, with defined hexagonal scutes confined to its surface. The reference prompt avoids a whole-image `patterned` description.
- Shared suffix: `Friendly high-quality 3D cartoon render, soft studio lighting, plain neutral gray background`.
- Common negatives: duplicate/multiple animals, multiple views, tiled composition, wallpaper, repeating motifs, split panels, character sheet, collage, cropped body, pedestal, text, watermark.
- Hare-specific negatives: pink/red vest, bow tie, necktie, shirt, trousers, hat, accessories. Pink ears/nose remain allowed; the entire color pink is not negated.
- Tortoise-specific negatives: green shell, repeated tortoises, mosaic, flat illustration, clothing. Green skin remains allowed.

SDXL reference size is explicitly 1024x1024 for v3, following the [pinned Diffusers recommendation](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/sdxl#text-to-image). This does not establish that 768 caused the old failure. Existing general workers keep their 768 default. The first trial uses 30 steps and the unchanged guidance 6.5. No refiner, new checkpoint, adapter, training or additional control model is added for references.

## Exact Colab actions: two images only

1. Download [character_references_v3.zip](https://raw.githubusercontent.com/SandeshAwate6055/quiet-meteor-73/master/examples/character_references_v3.zip). Open [the notebook in Colab](https://colab.research.google.com/github/SandeshAwate6055/quiet-meteor-73/blob/master/notebooks/phase2_batch_worker.ipynb), save a **fresh** copy to Drive, and start with a CPU runtime. Do not use a previously imported v2 notebook/session.
2. Leave default `MODE="dry_run"`. Run numbered cells 1-3; upload that ZIP once in cell 2. Expect `required_reference_ids=["hare","tortoise"]`, exactly **2 new character images**, **0 scenes**, **0 reused images**, and no blockers. Save the printed `PACKAGE_PATH`. Cell 4 in dry-run mode also performs no inference.
3. Manually select **Runtime > Change runtime type > GPU**. In controls, fill `PACKAGE_PATH` with the saved Drive ZIP path, leave `RESULT_BATCH_ID=""` and `REFERENCE_BATCH_MANIFEST=""`, then set:

   ```python
   MODE = "references"
   MAX_NEW_IMAGES = 2
   RUN_GENERATION = True
   INSTALL_COLAB_DEPS = True
   LICENSES_REVIEWED = True  # Read the linked checkpoint license first
   FAMILY = "sdxl"
   STEPS = 30
   BASE_SEED = 20261124
   IP_ADAPTER_SCALE = 0.35  # unused for these text-only reference jobs
   ALLOW_UNAPPROVED_REFERENCE_PREVIEW = False
   REFERENCES_REVIEWED = False
   QUALITY_REVIEWED = False
   ```

4. Run numbered cells 1-5. The existing backend uses `stabilityai/stable-diffusion-xl-base-1.0`, fp16, CPU offload, 1024x1024, CFG **6.5**, one image per character. The unchanged stable-ID formula gives hare seed **3839116770**, tortoise seed **595849088**. The actual checkpoint commit and runtime versions are resolved/recorded in the manifest and pinned on resume. Reference generation does not load IP-Adapter weights. The adapter configuration remains available for future scene requests only.
5. Inspect the contact sheet and full-size PNGs against the criteria above. Both must look like one compatible friendly 3D cast. Technical `complete` means all requested files exist and validate; it is **not** visual approval. Run cell 6 to export, then disconnect/delete the GPU runtime. **Stop: no challenge, smoke or full batch.** Their modes are blocked by this request even if selected accidentally.

The ZIP contains its exact request, checksum index and shared worker source, with an empty reference index. The fresh-reference policy excludes external/historical candidates. A successful image from the **same exact** request/settings/code result directory resumes after an interruption; no successful reference is regenerated automatically. Each image is saved to Drive immediately. A failure is explicit and cannot produce a false complete status.

## Retry only one visually rejected reference

A valid PNG that fails human review is still a technical success, so rerunning its old batch correctly reuses it. To get a new candidate, create a **new** request/attempt for the rejected character:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_reference_v3 --characters tortoise --attempt 2 --name tortoise_reference_v3_retry2 --output-dir outputs/reference_retry_v3
```

For hare, replace both occurrences of `tortoise` with `hare`. For a later retry, increment `--attempt` and choose a new `--name`. The printed base seed is `20261123 + attempt`. The builder never overwrites a different existing ZIP/JSON. To reproduce the initial package from this exact code checkout:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_reference_v3
```

Open a fresh Colab session/copy, upload only the new single-character ZIP, keep `RESULT_BATCH_ID=""` and `REFERENCE_BATCH_MANIFEST=""`, set `BASE_SEED=20261125` for attempt 2 and `MAX_NEW_IMAGES=1`. Repeat CPU dry run, then the explicit reference settings above. Expect only that character job, zero scenes, zero reference reuse. For attempt 2 the tortoise seed is **595849089** (hare **3839116771** if selected). The acceptable other character stays unchanged in its original Drive batch; it is not requested or regenerated. Changed requests/settings/code always have a new generation identity, and pointing at an incompatible old directory fails.

To resume a **technical interruption**, instead use the same ZIP, settings, seed and result directory. Preflight will show only genuinely missing references. Do not increment attempts merely to resume an interrupted job.

## Local validation/import after human review

After exporting, extract to a new folder and use the existing APIs (candidate import does not approve):

```powershell
$referenceManifest = Read-Host 'Full path to the NEW extracted manifest.json'
.\.venv\Scripts\python.exe -c "import sys; from src.batch_handoff import validate_result; print(validate_result(sys.argv[1]))" $referenceManifest
.\.venv\Scripts\python.exe -m src.assets import-batch $referenceManifest
.\.venv\Scripts\python.exe -m src.assets review --output-dir outputs/reference_v3_review
```

Reference-only handoffs are standalone quality requests, not registered production-cache requests. Use the existing asset import, not production intake's request-binding check. Checksums, provenance, duplicate-safe candidate IDs and manual approval locks are preserved. No reference was imported or approved as part of preparing v3. Keep v2 files and previous videos untouched. A later task can prepare `challenge` only after human reference review; no scene-generation authorization is implied here.
