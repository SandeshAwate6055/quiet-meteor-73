"""Reference-only v3 controls. Synthetic images/mock inference; no GPU packages."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
import zipfile

from PIL import Image

from scripts.prepare_reference_v3 import prepare, ROOT
from src.asset_store import AssetStore
from src.assets import import_batch
from src.batch_handoff import validate_result
from src.characters import load_bible, reference_prompt, reference_negative_prompt, REFERENCE_SPEC_REVISION
from src.colab_images import ColabDiffusersBackend
from src.image_batch import BatchSettings, BatchWorker, reference_sources, scene_seed, validate_request
from src.production_state import file_hash, write_json
from src.schema import load_story
from test_image_batch import SyntheticBackend


class ReferenceGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bible = load_bible(ROOT / "configs/character_bible.json")
        self.settings = BatchSettings(base_seed=20261124, steps=30, reference_size=1024)

    def worker(self, **kwargs):
        artifacts = prepare(self.root / "handoff")
        return BatchWorker(artifacts["request_path"], self.root / "result", settings=self.settings, **kwargs)

    def test_precise_reference_subjects_shared_style_and_specific_negatives(self):
        hare, tortoise = (self.bible.characters[c] for c in ("hare", "tortoise"))
        prompts = [reference_prompt(c, self.bible) for c in (hare, tortoise)]
        for prompt in prompts:
            for text in ("Exactly one", "full-body", "three-quarter", "entire body visible"):
                self.assertIn(text, prompt.lower() if text == "full-body" else prompt)
            self.assertTrue(prompt.endswith(self.bible.metadata["reference_style"] + "."))
            self.assertNotIn("forest", prompt)
            self.assertLess(len(prompt.split()), 60)  # Actual CLIP counts checked in Colab, not word counts.
        for text in ("white rabbit", "royal blue sleeveless vest", "Pink inner ears", "small pink nose"):
            self.assertIn(text, prompts[0])
        for text in ("green tortoise", "rounded brown back shell clearly visible", "hexagonal scutes confined to that shell"):
            self.assertIn(text, prompts[1])
        self.assertNotIn("patterned", prompts[1])  # Do not describe the whole composition as a pattern.
        for character in (hare, tortoise):
            negative = reference_negative_prompt(character)
            for text in ("duplicate animals", "multiple views", "tiled composition", "wallpaper",
                         "repeating motifs", "split panels", "character sheet", "text", "watermark"):
                self.assertIn(text, negative)
            self.assertNotIn("pink", [s.strip() for s in negative.split(",")])
            self.assertNotIn("green", [s.strip() for s in negative.split(",")])
        self.assertIn("pink vest", reference_negative_prompt(hare))
        self.assertIn("bow tie", reference_negative_prompt(hare))
        self.assertIn("green shell", reference_negative_prompt(tortoise))

    def test_new_package_zero_scenes_no_reference_pixels_reproducible_and_non_overwriting(self):
        artifacts = prepare(self.root)
        with zipfile.ZipFile(artifacts["package_path"]) as archive:
            request = json.loads(archive.read("request.json"))
            self.assertEqual(request["scenes"], [])
            self.assertEqual(request["generation_policy"]["required_reference_ids"], ["hare", "tortoise"])
            self.assertEqual(request["reference_prompt_revision"], REFERENCE_SPEC_REVISION)
            self.assertEqual(json.loads(archive.read("references.json"))["references"], [])
            self.assertFalse(any(name.endswith(".png") for name in archive.namelist()))
            self.assertIn("src/colab_images.py", archive.namelist())
        self.assertEqual(prepare(self.root)["package_sha256"], artifacts["package_sha256"])
        original = Path(artifacts["package_path"]).read_bytes()
        with self.assertRaisesRegex(ValueError, "nothing was overwritten"):
            prepare(self.root, attempt=2)
        self.assertEqual(Path(artifacts["package_path"]).read_bytes(), original)

    def test_dry_run_plans_only_two_refs_and_scene_modes_are_blocked(self):
        worker = self.worker()
        factory = MagicMock()
        plan = worker.execute(factory, mode="dry_run", max_new_images=2)
        self.assertEqual(plan["new_images"], [{"kind": "character", "id": c} for c in ("hare", "tortoise")])
        self.assertEqual(plan["requested_scene_ids"], [])
        self.assertEqual(plan["blockers"], [])
        self.assertFalse(worker.root.exists())
        for mode in ("smoke", "full"):
            with self.assertRaisesRegex(ValueError, "Reference-only"):
                worker.execute(factory, mode=mode, max_new_images=2, quality_reviewed=True)
        factory.assert_not_called()

    def test_external_failed_candidates_never_reused_but_preserved(self):
        image = self.root / "failed.png"
        Image.new("RGB", (128, 128), "pink").save(image)
        checksum = file_hash(image)
        index = self.root / "references.json"
        # Even an external v3 candidate cannot silently satisfy a fresh request.
        write_json(index, {"references": [{"target_id": "hare", "file": image.name, "sha256": checksum,
                    "metadata": {"reference_spec_revision": REFERENCE_SPEC_REVISION}}]})
        worker = self.worker(references=reference_sources(index))
        plan = worker.preflight(mode="references", max_new_images=2)
        self.assertEqual(plan["new_image_count"], 2)
        self.assertFalse(plan["reused_images"])
        self.assertEqual(file_hash(image), checksum)

    def test_partial_reference_failure_resume_and_idempotence(self):
        worker = self.worker()
        backend = SyntheticBackend()
        backend.fail_id = "tortoise"
        data = worker.execute(lambda *args: backend, mode="references", max_new_images=2)
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["completion"]["missing_reference_ids"], ["tortoise"])
        self.assertEqual(data["completion"]["failed_reference_ids"], ["tortoise"])
        self.assertEqual(validate_result(worker.manifest_path)["completed_reference_ids"], ["hare"])
        hare = data["assets"][0]
        path = worker.root / hare["file"]
        before = (file_hash(path), path.stat().st_mtime_ns)
        backend.fail_id = None
        resumed = self.worker()
        self.assertEqual(resumed.preflight(mode="references", max_new_images=1)["new_image_count"], 1)
        data = resumed.execute(lambda *args: backend, mode="references", max_new_images=1)
        self.assertEqual(data["status"], "complete")
        self.assertEqual([c[1] for c in backend.calls], ["hare", "tortoise", "tortoise"])
        self.assertEqual((file_hash(path), path.stat().st_mtime_ns), before)
        self.assertEqual(len(data["assets"]), 2)
        self.assertTrue(all(a["metadata"]["resolution"] == [1024, 1024] for a in data["assets"]))
        self.assertNotIn("smoke_quality_review", data)
        factory = MagicMock(side_effect=AssertionError("No model should load on a complete rerun"))
        self.worker().execute(factory, mode="references", max_new_images=0)
        factory.assert_not_called()
        self.assertEqual(validate_result(worker.manifest_path)["missing_reference_ids"], [])

    def test_reference_only_request_rejects_scenes_and_missing_required_ids(self):
        for invalid in (True, 1024.0, 999):
            with self.assertRaisesRegex(ValueError, "reference_size"):
                BatchSettings(reference_size=invalid).validate()
        worker = self.worker()
        request = json.loads(worker.request_path.read_text())
        request["scenes"] = [{"id": "challenge", "prompt": "one scene", "characters": ["hare"]}]
        with self.assertRaisesRegex(ValueError, "scope"):
            validate_request(request)
        request["scenes"] = []
        request["generation_policy"]["required_reference_ids"] = ["hare"]
        with self.assertRaisesRegex(ValueError, "required_reference_ids"):
            validate_request(request)

    def test_v3_import_remains_unapproved_and_duplicate_safe(self):
        worker = self.worker()
        worker.execute(lambda *args: SyntheticBackend(), mode="references", max_new_images=2)
        validate_result(worker.manifest_path)
        store = AssetStore(self.root / "library")
        story = load_story(ROOT / "scripts/tortoise_and_hare.json")
        first = import_batch(store, worker.manifest_path, story, self.bible)
        second = import_batch(store, worker.manifest_path, story, self.bible)
        self.assertEqual(first, second)
        self.assertEqual(len(store.read()["assets"]), 2)
        for identity in ("hare", "tortoise"):
            self.assertIsNone(store.approved("character", identity))
        self.assertTrue(all(a["metadata"]["reference_spec_revision"] == REFERENCE_SPEC_REVISION
                            for a in store.read()["assets"].values()))

    def test_missing_reference_cannot_be_forged_complete(self):
        worker = self.worker()
        backend = SyntheticBackend()
        backend.fail_id = "tortoise"
        data = worker.execute(lambda *args: backend, mode="references", max_new_images=2)
        data["status"] = "complete"
        write_json(worker.manifest_path, data)
        with self.assertRaisesRegex(ValueError, "required references are missing/failed"):
            validate_result(worker.manifest_path)

    def test_one_rejected_reference_gets_new_request_seed_and_batch_without_hare_job(self):
        original = self.worker()
        backend = SyntheticBackend()
        original.execute(lambda *args: backend, mode="references", max_new_images=2)
        hare = next(a for a in original.data["assets"] if a["target_id"] == "hare")
        hare_path = original.root / hare["file"]
        previous_bytes = hare_path.read_bytes()
        retry = prepare(self.root / "retry", name="tortoise_reference_v3_retry2", characters=["tortoise"], attempt=2)
        settings = BatchSettings(base_seed=20261125, steps=30, reference_size=1024)
        worker = BatchWorker(retry["request_path"], self.root / "retry_result", settings=settings)
        self.assertNotEqual(worker.request_hash, original.request_hash)
        self.assertNotEqual(worker.generation_identity, original.generation_identity)
        self.assertNotEqual(scene_seed("tortoise", settings.base_seed), scene_seed("tortoise", self.settings.base_seed))
        model = SyntheticBackend()
        data = worker.execute(lambda *args: model, mode="references", max_new_images=1)
        self.assertEqual([c[1] for c in model.calls], ["tortoise"])
        self.assertEqual(data["completion"]["completed_reference_ids"], ["tortoise"])
        self.assertEqual(hare_path.read_bytes(), previous_bytes)
        self.assertEqual(validate_result(worker.manifest_path)["asset_count"], 1)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            BatchWorker(retry["request_path"], self.root / "wrong_seed", settings=self.settings)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            BatchWorker(original.request_path, original.root, settings=BatchSettings(base_seed=20261124, steps=31, reference_size=1024))

    def test_reference_backend_passes_distinct_prompts_without_adapter_and_keeps_settings(self):
        backend = ColabDiffusersBackend.__new__(ColabDiffusersBackend)
        backend.settings, backend.torch = self.settings, MagicMock()
        backend.provenance = {"adapter_id": "h94/IP-Adapter", "adapter_revision": "fixture"}
        backend.torch.cuda.max_memory_allocated.return_value = 0
        backend.pipe = MagicMock()
        tokenizer = MagicMock(model_max_length=77)
        tokenizer.side_effect = lambda text, truncation: {"input_ids": list(range(len(text.split()) + 2))}
        backend.pipe.tokenizer = backend.pipe.tokenizer_2 = tokenizer
        backend.pipe.return_value = SimpleNamespace(images=[Image.new("RGB", (1024, 1024))], nsfw_content_detected=None)
        backend.adapter_loaded = False
        backend._adapter = MagicMock()
        character = self.bible.characters["hare"]
        job = {"kind": "character", "prompt": reference_prompt(character, self.bible),
               "negative_prompt": reference_negative_prompt(character), "seed": 3839116770, "size": (1024, 1024)}
        _, metadata = backend.generate(job, {})
        options = backend.pipe.call_args.kwargs
        self.assertEqual(options["prompt"], job["prompt"])
        self.assertEqual(options["negative_prompt"], job["negative_prompt"])
        self.assertEqual(options["num_inference_steps"], 30)
        self.assertEqual(options["guidance_scale"], 6.5)
        self.assertNotIn("ip_adapter_image", options)
        self.assertEqual(metadata["conditioning"], "text_only")
        self.assertIsNone(metadata["adapter_id"])
        backend._adapter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
