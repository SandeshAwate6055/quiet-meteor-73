# Current handoff: character references v3 only

Use [character_references_v3.zip](character_references_v3.zip) and [its readable request](character_references_v3.json) with the updated batch notebook. [Exact Colab steps, reference settings, diagnosis and single-character retry commands](../notebooks/REFERENCE_V3.md).

This small reproducible package has two character specifications, **zero scenes**, an empty reference index and checksummed worker source. No historical candidates, generated images, credentials, models or local machine settings are included. Both v2 references failed visual review and remain unapproved; all old files are preserved. V3 is locally tested but **not GPU-executed**. The current task stops after reference review, without challenge generation.

Reproduce from this exact code checkout with `.\.venv\Scripts\python.exe -m scripts.prepare_reference_v3`. The builder refuses to overwrite different existing files. Use `--characters tortoise --attempt 2 --name tortoise_reference_v3_retry2 --output-dir outputs/reference_retry_v3` to prepare a new one-reference retry without scheduling hare. Use the printed new base seed and a fresh result identity.

Everything below describes historical packages; their scene-generation steps are superseded by the v3 reference-only quality gate. Do not run them for this task.

## Historical challenge request (v2): superseded

Use `challenge_smoke_v2.zip` with the updated batch notebook. `challenge_smoke_v2.json` is its readable request. The original `colab_handoff.zip` and its historical references are retained unchanged.

The v2 ZIP below is retained byte-for-byte as history. The old builder command is shown for context; current code produces different content and refuses to overwrite that historical package. Use the v3 builder instead:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_challenge_smoke
```

The builder uses existing request export/handoff APIs and refuses to overwrite a different existing package. The request contains **only `challenge`**, fresh single-view reference specifications, separate negative prompts, masked-reference conditioning, a failed-previous-smoke note and a policy disabling full batches. No model weights, credentials, historical PNGs, local paths or generated scene are included. New reference candidates must be made on Colab, not selected from the old character sheet.

Open `notebooks/phase2_batch_worker.ipynb` in a fresh Colab/Drive copy. On CPU run cells 1–3 with `MODE="dry_run"`; upload this v2 ZIP and keep the printed `PACKAGE_PATH`. Leave `RESULT_BATCH_ID=""` so a new content/config/code identity is used.

For reference generation (only when you intentionally choose a GPU runtime):

```python
MODE = "references"
MAX_NEW_IMAGES = 2
RUN_GENERATION = True
INSTALL_COLAB_DEPS = True
LICENSES_REVIEWED = True
REFERENCES_REVIEWED = False
QUALITY_REVIEWED = False
ALLOW_UNAPPROVED_REFERENCE_PREVIEW = False
```

Run cells 1–5. Inspect the two NEW candidates before spending another image: Hare must be one full-body 3/4 white rabbit, pink inner ears, blue vest; Tortoise must be one full-body 3/4 side view with green skin and a clearly visible rounded brown patterned back shell. No pedestal, duplicate figures, front/back sheet or split panels. Failed references require a revised/new batch; keep all old files.

Only after both pass human reference review, keep the same saved package and result folder and change:

```python
MODE = "smoke"
MAX_NEW_IMAGES = 1
REFERENCES_REVIEWED = True
ALLOW_UNAPPROVED_REFERENCE_PREVIEW = True
QUALITY_REVIEWED = False
```

Run cells 1–5 again. This uses the two new (still unapproved) reference candidates for experimental preview and generates only `challenge`. It does not approve anything in the local asset library. The result must be exactly two correct animals at one sunny forest race starting line in one unified cinematic frame. If it fails, stop and retain it as a failed visual candidate. Do not schedule the remaining four scenes. Export with cell 6 and disconnect/delete the GPU runtime when finished.

The previous smoke's three figures/panels are consistent with a global IP-Adapter input containing a two-view rabbit plus another reference. The code did combine those pixels into a 512-square reference board; that is a verified implementation fact. The individual cause of color drift or each artifact cannot be proved without an ablation. New inputs avoid that composition, use separate masked embeddings at scale 0.35 and shorter subject-first prompts. Masks guide regions but do not guarantee independent animal identities or exact layouts. This revision has **not** been executed on GPU.

The underlying import/approval controls and version-1 result envelope remain compatible; new results carry worker-version 2, generation identity, settings, reference hashes, mask layout/hashes, prompts and actual tokenizer counts. The original deterministic scene seed formula is unchanged. No human scene-quality acceptance is inferred from a completed generation job.

For local candidate review after exporting, extract the result ZIP to a new folder and use the existing asset commands:

```powershell
$smokeManifest = Read-Host 'Full path to the NEW extracted smoke manifest.json'
.\.venv\Scripts\python.exe -m src.assets import-batch $smokeManifest
.\.venv\Scripts\python.exe -m src.assets review --output-dir outputs/challenge_v2_review
```

This is a standalone quality-control request, not a registered production-cache request. Candidate import/review leaves all approvals unchanged. Keep the failed smoke and old references unapproved; production selection changes only after your separate explicit approval of acceptable candidates. The current milestone stops at reviewing `challenge`.

## Historical public handoff (v1)

`colab_handoff.zip` is the audited public version of the existing small local request package. It is intended for the dry run and an explicitly started one-scene smoke test in `notebooks/phase2_batch_worker.ipynb`. It contains no model weights, credentials, videos or private datasets.

Contents:

- `request.json`: all five Tortoise and Hare scene prompts and the Character Bible prompt data.
- `references.json` and two PNGs: the existing Rabbit and Tortoise reference candidates, both explicitly **unapproved**.
- `src/`: the same shared batch-worker modules used by the project.
- `bundle_index.json`: checksums for every payload file.

The original generated reference PNG bytes, prompts, seeds, checkpoint IDs/revisions, licenses and generation settings are preserved. Copied metadata omits absolute local paths; original local files are unchanged. These generated cartoon illustrations are review candidates, not guaranteed consistent character designs. Rabbit has two views and a teal-blue vest; Tortoise's back shell is not clearly visible. Review their suitability before spending GPU time.

Open the batch notebook from this repository in Google Colab, save a copy to Drive, and begin on a **CPU runtime**. Run cells 1–3 with `MODE="dry_run"`; upload `colab_handoff.zip` when cell 2 prompts. Preflight lists five scene jobs and the two references to reuse, with explicit consent blockers because the references are unapproved. Do not install GPU packages or start inference for the dry run.

For an intentional experimental smoke test, manually choose a GPU runtime, keep the saved `PACKAGE_PATH`, and set `MODE="smoke"`, `MAX_NEW_IMAGES=1`, `RUN_GENERATION=True`, `INSTALL_COLAB_DEPS=True`, `LICENSES_REVIEWED=True`, and `ALLOW_UNAPPROVED_REFERENCE_PREVIEW=True`. Read the model licenses linked in the notebook before enabling generation. Run through cell 5, then review the scene. This consent does not approve any asset for production. After successful quality review, full mode can generate the four remaining scenes with the same persistent references/result folder.

GPU image generation has not been executed as part of publication. No automatic Colab runtime control or paid API is provided. The notebook contains the remaining export/import/approval steps, and the main README documents the master production command and tests.
