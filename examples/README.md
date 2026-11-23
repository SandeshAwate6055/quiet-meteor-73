# Portable Colab handoff

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
