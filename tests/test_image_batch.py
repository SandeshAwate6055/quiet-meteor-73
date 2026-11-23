"""Batch orchestration tested with an explicitly synthetic backend, never a model."""
import ast
import base64
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from src.asset_store import AssetStore, digest
from src.assets import export_request
from src.batch_handoff import create_handoff, receive_result, validate_result
from src.characters import load_bible
from src.image_batch import BatchWorker, BatchSettings, reference_sources, scene_seed, contact_sheet
from src.production import ROOT, produce
from src.production_state import file_hash, write_json
from src.schema import load_story
from src.media import find_tool
from src.narration import NarrationResult, write_silence


class SyntheticBackend:
    provenance = {"model_id": "synthetic-test-fixture", "model_revision": "fixture-v1"}

    def __init__(self, fail_id=None, interrupt_id=None, bad_conditioning=False):
        self.calls = []
        self.fail_id, self.interrupt_id, self.bad_conditioning = fail_id, interrupt_id, bad_conditioning

    def generate(self, job, refs):
        self.calls.append((job["kind"], job["id"], job["seed"]))
        if job["id"] == self.fail_id:
            raise RuntimeError("Synthetic adapter/generation failure")
        if job["id"] == self.interrupt_id:
            raise KeyboardInterrupt("Synthetic interrupted runtime")
        condition = "text_only" if job["kind"] == "character" else "ip_adapter_contact_sheet" if refs else "not_applicable"
        if self.bad_conditioning:
            condition = "text_only"
        return Image.new("RGB", job["size"], "#f2ab59"), {**self.provenance, "conditioning": condition}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = AssetStore(self.root / "library")
        self.story = load_story(ROOT / "scripts/tortoise_and_hare.json")
        self.bible = load_bible(ROOT / "configs/character_bible.json")
        self.request = self.root / "request.json"
        export_request(self.store, self.story, self.bible, self.request, [s.id for s in self.story.scenes])
        rows = []
        for identity in ("hare", "tortoise"):
            path = self.root / f"{identity}.png"
            Image.new("RGB", (128, 128), "white" if identity == "hare" else "green").save(path)
            rows.append({"target_id": identity, "file": path.name, "sha256": file_hash(path),
                         "asset_id": f"fixture_{identity}", "metadata": {"model_id": "synthetic-original-reference"}})
        self.ref_index = self.root / "references.json"
        write_json(self.ref_index, {"version": 1, "references": rows})
        self.sources = reference_sources(self.ref_index)
        self.result_dir = self.root / "result"
        self.backend = SyntheticBackend()
        self.loads = 0

    def factory(self, settings, previous):
        self.loads += 1
        return self.backend

    def worker(self, sources=True):
        return BatchWorker(self.request, self.result_dir, references=self.sources if sources else [])

    def smoke(self):
        return self.worker().execute(self.factory, mode="smoke", allow_unapproved_references=True, max_new_images=1)

    def test_full_five_scene_preflight_reuses_references_without_model_loading(self):
        plan = self.worker().execute(self.factory, mode="dry_run", allow_unapproved_references=True, max_new_images=5)
        self.assertEqual(plan["requested_scene_ids"], [s.id for s in self.story.scenes])
        self.assertEqual(plan["new_image_count"], 5)
        self.assertEqual([r["id"] for r in plan["reused_images"]], ["hare", "tortoise"])
        self.assertEqual(self.loads, 0)
        self.assertFalse(self.result_dir.exists())

    def test_unapproved_reference_isolation_and_full_quality_gate(self):
        plan = self.worker().preflight(mode="smoke", max_new_images=1)
        self.assertTrue(any("Unapproved reference" in b for b in plan["blockers"]))
        with self.assertRaisesRegex(ValueError, "Preflight blocked"):
            self.worker().execute(self.factory, mode="smoke", max_new_images=1)
        with self.assertRaisesRegex(ValueError, "smoke scene"):
            self.worker().execute(self.factory, mode="full", quality_reviewed=True,
                                  allow_unapproved_references=True, max_new_images=5)
        self.assertEqual(self.loads, 0)

    def test_approved_embedded_references_need_no_preview_opt_in(self):
        request = json.loads(self.request.read_text())
        for row in request["characters"]:
            payload = (self.root / f"{row['id']}.png").read_bytes()
            row.update(approved_reference_png_base64=base64.b64encode(payload).decode(),
                       approved_reference_sha256=digest(payload), approved_asset_id=f"approved_fixture_{row['id']}")
        write_json(self.request, request)
        result = self.worker(sources=False).execute(self.factory, mode="smoke", max_new_images=1)
        scene = next(a for a in result["assets"] if a["kind"] == "scene")
        self.assertFalse(scene["metadata"]["experimental_preview"])
        self.assertEqual(len(self.backend.calls), 1)

    def test_smoke_then_full_resumes_and_cannot_claim_incomplete_batch_complete(self):
        stale_worker = self.worker()
        first = self.smoke()
        refreshed = stale_worker.preflight(mode="full", quality_reviewed=True, allow_unapproved_references=True, max_new_images=4)
        self.assertEqual(refreshed["new_image_count"], 4)
        self.assertEqual(first["status"], "awaiting_smoke_review")
        self.assertEqual(len(first["completion"]["missing_scene_ids"]), 4)
        paths = {a["file"]: file_hash(self.result_dir / a["file"]) for a in first["assets"]}
        second = self.worker().execute(self.factory, mode="full", quality_reviewed=True,
                                       allow_unapproved_references=True, max_new_images=4)
        self.assertEqual(second["status"], "complete")
        self.assertEqual(second["completion"]["missing_scene_ids"], [])
        self.assertEqual(len(self.backend.calls), 5)
        self.assertTrue(all(kind == "scene" for kind, _, _ in self.backend.calls))
        for relative, checksum in paths.items():
            self.assertEqual(file_hash(self.result_dir / relative), checksum)
        self.assertEqual(validate_result(self.result_dir / "manifest.json")["completed_scene_ids"], [s.id for s in self.story.scenes])
        before_loads = self.loads
        self.worker().execute(self.factory, mode="full", quality_reviewed=True,
                              allow_unapproved_references=True, max_new_images=0)
        self.assertEqual(self.loads, before_loads)

    def test_failed_and_interrupted_jobs_are_persisted_and_resumable(self):
        self.smoke()
        self.backend.fail_id = "nap"
        failed = self.worker().execute(self.factory, mode="full", quality_reviewed=True,
                                      allow_unapproved_references=True, max_new_images=4)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["completion"]["failed_scene_ids"], ["nap"])
        report = validate_result(self.result_dir / "manifest.json")
        self.assertIn("racing_ahead", report["completed_scene_ids"])
        successful_calls = len(self.backend.calls)
        self.backend.fail_id = None
        self.backend.interrupt_id = "steady_steps"
        with self.assertRaises(KeyboardInterrupt):
            self.worker().execute(self.factory, mode="full", quality_reviewed=True,
                                  allow_unapproved_references=True, max_new_images=3)
        interrupted = json.loads((self.result_dir / "manifest.json").read_text())
        self.assertEqual(interrupted["status"], "interrupted")
        self.backend.interrupt_id = None
        resumed = self.worker().execute(self.factory, mode="full", quality_reviewed=True,
                                       allow_unapproved_references=True, max_new_images=2)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(sum(identity == "racing_ahead" for _, identity, _ in self.backend.calls), 1)
        self.assertGreater(len(self.backend.calls), successful_calls)

    def test_explicit_failure_when_reference_conditioning_is_not_applied(self):
        self.backend.bad_conditioning = True
        result = self.smoke()
        self.assertEqual(result["status"], "failed")
        self.assertIn("required reference conditioning", result["errors"][-1]["error"])
        self.assertEqual(result["completion"]["completed_scene_ids"], [])

    def test_generate_only_truly_missing_reference_then_reuse_it(self):
        self.sources = self.sources[:1]
        first = self.worker().execute(self.factory, mode="smoke", allow_unapproved_references=True, max_new_images=2)
        self.assertEqual(first["status"], "awaiting_smoke_review")
        self.assertEqual([kind for kind, _, _ in self.backend.calls], ["character", "scene"])
        self.assertEqual(self.backend.calls[0][1], "tortoise")
        plan = self.worker().preflight(mode="full", quality_reviewed=True, allow_unapproved_references=True, max_new_images=4)
        self.assertTrue(all(r["kind"] == "scene" for r in plan["new_images"]))

    def test_manifest_checksums_completeness_seeds_and_immutable_settings(self):
        self.smoke()
        report = validate_result(self.result_dir / "manifest.json")
        self.assertEqual(len(report["missing_scene_ids"]), 4)
        manifest = self.result_dir / "manifest.json"
        data = json.loads(manifest.read_text())
        data["status"] = "complete"
        write_json(manifest, data)
        with self.assertRaisesRegex(ValueError, "claims complete"):
            validate_result(manifest)
        data["status"] = "awaiting_smoke_review"
        scene = next(a for a in data["assets"] if a["kind"] == "scene")
        scene["metadata"]["seed"] += 1
        write_json(manifest, data)
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_result(manifest)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            BatchWorker(self.request, self.result_dir, references=self.sources, settings=BatchSettings(steps=12))

    def test_deterministic_seeds_do_not_depend_on_scene_selection_order(self):
        before = {s.id: scene_seed(s.id) for s in self.story.scenes}
        self.assertEqual(before, {s.id: scene_seed(s.id) for s in reversed(self.story.scenes)})
        self.smoke()
        scene = next(a for a in json.loads((self.result_dir / "manifest.json").read_text())["assets"] if a["kind"] == "scene")
        self.assertEqual(scene["metadata"]["seed"], before["challenge"])

    def test_package_contains_shared_code_and_keeps_candidates_unapproved(self):
        for identity in ("hare", "tortoise"):
            self.store.import_image(self.root / f"{identity}.png", "character", identity)
        package = self.root / "handoff.zip"
        result = create_handoff(self.request, self.store, package)
        again = create_handoff(self.request, self.store, self.root / "same_package.zip")
        self.assertEqual(again["package_sha256"], result["package_sha256"])
        self.assertEqual(result["reference_count"], 2)
        with zipfile.ZipFile(package) as archive:
            index = json.loads(archive.read("bundle_index.json"))
            for relative, checksum in index["files"].items():
                self.assertEqual(digest(archive.read(relative)), checksum)
            self.assertIn("src/image_batch.py", index["files"])
            references = json.loads(archive.read("references.json"))["references"]
            self.assertTrue(all(r["approval"] == "candidate" for r in references))
        self.assertFalse(self.store.read()["approvals"]["character"])

    def test_existing_import_review_workflow_is_duplicate_safe_and_preserves_provenance(self):
        self.smoke()
        production = self.root / "production"
        production.mkdir()
        write_json(production / "production_manifest.json", {"kind": "story_studio_production", "version": 1,
            "input": {"script_path": str(ROOT / "scripts/tortoise_and_hare.json"),
                      "bible_path": str(ROOT / "configs/character_bible.json"), "asset_directory": str(self.store.root)},
            "ai_generation_request": {"sha256": file_hash(self.request)}})
        first = receive_result(production, self.result_dir / "manifest.json")
        before = self.store.read()
        again = receive_result(production, self.result_dir / "manifest.json")
        self.assertTrue(again["duplicate_result"])
        self.assertEqual(self.store.read(), before)
        self.assertEqual(len(before["assets"]), 3)
        self.assertEqual(before["approvals"], {"scene": {}, "character": {}})
        self.assertEqual(len(first["missing_scene_ids"]), 4)
        self.assertTrue(Path(first["review_pages"][0]).is_file())
        scene = next(r for r in before["assets"].values() if r["kind"] == "scene")
        self.assertTrue(scene["metadata"]["experimental_preview"])
        self.assertIn("source_batch", scene["metadata"])
        self.assertTrue(contact_sheet(self.result_dir).exists())
        export = self.root / "export.zip"
        with zipfile.ZipFile(export, "w") as archive:
            for source in self.result_dir.rglob("*"):
                if source.is_file():
                    archive.write(source, source.relative_to(self.result_dir).as_posix())
        self.assertTrue(receive_result(production, export)["duplicate_result"])
        self.assertEqual(self.store.read(), before)
        malicious = self.root / "bad_paths.zip"
        with zipfile.ZipFile(malicious, "w") as archive:
            archive.writestr("../outside.png", b"unsafe")
        with self.assertRaises(ValueError):
            receive_result(production, malicious)
        self.assertFalse((production / "outside.png").exists())
        # A request may become historical after approving the smoke image and
        # replanning the remaining scenes. Its remaining results must still import.
        production_data = json.loads((production / "production_manifest.json").read_text())
        production_data["cache"] = {"old_request": {"name": "ai_request", "status": "completed", "result": {"sha256": file_hash(self.request)}}}
        production_data["ai_generation_request"]["sha256"] = "newer_request"
        write_json(production / "production_manifest.json", production_data)
        self.assertFalse(receive_result(production, self.result_dir / "manifest.json")["request_is_current"])

    def test_budget_and_damaged_reference_fail_before_backend_loading(self):
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            self.worker().execute(self.factory, mode="smoke", allow_unapproved_references=True, max_new_images=0)
        self.assertEqual(self.loads, 0)
        (self.root / "hare.png").write_bytes(b"damaged")
        with self.assertRaises((ValueError, OSError)):
            self.worker().execute(self.factory, mode="smoke", allow_unapproved_references=True, max_new_images=1)
        self.assertEqual(self.loads, 0)

    def test_damaged_result_is_rejected_before_any_asset_import(self):
        result = self.smoke()
        scene = next(a for a in result["assets"] if a["kind"] == "scene")
        (self.result_dir / scene["file"]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_result(self.result_dir / "manifest.json")
        self.assertFalse(self.store.read()["assets"])

    def test_import_then_approve_rebuilds_only_related_scene_stages(self):
        try:
            ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
        except RuntimeError as exc:
            self.skipTest(str(exc))
        raw = json.loads((ROOT / "scripts/tortoise_and_hare.json").read_text())
        raw["scenes"] = raw["scenes"][:2]
        for row in raw["scenes"]:
            row["duration"] = .4
        script = self.root / "short_story.json"
        write_json(script, raw)
        pipeline = json.loads((ROOT / "configs/pipeline.json").read_text())
        pipeline.update(fps=12, tail_seconds=0)
        pipeline_path = self.root / "pipeline.json"
        write_json(pipeline_path, pipeline)
        settings = self.root / "production_settings.json"
        write_json(settings, {"vertical_scene_ids": ["challenge"]})
        for identity in ("hare", "tortoise"):
            self.store.import_image(self.root / f"{identity}.png", "character", identity)
        calls = []

        class Voice:
            cache_key = "synthetic_voice_for_batch_integration"

            def synthesize(self, scene, directory, settings):
                calls.append(scene.id)
                path = directory / f"{scene.id}.wav"
                write_silence(path, .25)
                return NarrationResult(path, .25, "synthetic_fixture")

        output = self.root / "production_integration"

        def production():
            return produce(script, output, settings_path=settings, pipeline_path=pipeline_path,
                asset_dir=self.store.root, voice=Voice(), ffmpeg=ffmpeg, ffprobe=ffprobe)

        first = production()
        previous_output = first["outputs"]["landscape"]["path"]
        previous_hash = file_hash(previous_output)
        worker = BatchWorker(first["ai_generation_request"]["path"], self.result_dir, references=self.sources)
        worker.execute(self.factory, mode="smoke", allow_unapproved_references=True, max_new_images=1)
        receipt = receive_result(output, self.result_dir / "manifest.json")
        unapproved = production()
        self.assertEqual(unapproved["runs"][-1]["executed_stages"], ["ai_request"])
        self.assertEqual(unapproved["scenes"][0]["images"]["landscape"]["provider"], "pillow")
        scene_id = next(identity for identity in receipt["imported_asset_ids"] if identity.startswith("scene_challenge_"))
        self.store.approve(scene_id)  # Explicit approval of synthetic fixture in isolated temp library only.
        approved = production()
        self.assertEqual(approved["scenes"][0]["images"]["landscape"]["provider"], "approved_file")
        self.assertIn("clip:racing_ahead:landscape", approved["runs"][-1]["reused_stages"])
        self.assertEqual(calls, ["challenge", "racing_ahead"])
        self.assertEqual(file_hash(previous_output), previous_hash)

    def test_batch_notebook_is_unexecuted_and_defaults_to_dry_run(self):
        notebook = json.loads((ROOT / "notebooks/phase2_batch_worker.ipynb").read_text(encoding="utf-8"))
        values = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            compile(source, "batch_cell", "exec")
            for statement in ast.parse(source).body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    try:
                        values[statement.targets[0].id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        pass
        self.assertEqual(values["MODE"], "dry_run")
        self.assertFalse(values["RUN_GENERATION"])
        self.assertFalse(values["ALLOW_UNAPPROVED_REFERENCE_PREVIEW"])
        self.assertFalse(values["QUALITY_REVIEWED"])
        self.assertNotIn("USE_REFERENCE_CONDITIONING", values)


if __name__ == "__main__":
    unittest.main()
