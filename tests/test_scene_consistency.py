"""Prompt, spatial-mask and compatibility checks; no Torch/models are loaded."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from PIL import Image

from src.asset_store import AssetStore
from src.assets import export_request
from src.characters import (load_bible, scene_prompt, reference_prompt, PROMPT_REVISION,
                            SCENE_NEGATIVE_PROMPT, REFERENCE_NEGATIVE_PROMPT)
from src.colab_images import ColabDiffusersBackend
from src.image_batch import BatchSettings, BatchWorker, generation_identity, reference_sources, scene_seed
from src.image_conditioning import CONDITIONING_REVISION, reference_layout, reference_masks
from src.production_state import file_hash, write_json
from src.schema import load_story
from src.main import ROOT
from scripts.prepare_challenge_smoke import prepare
from test_image_batch import SyntheticBackend


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bible = load_bible(ROOT / "configs/character_bible.json")
        self.story = load_story(ROOT / "scripts/tortoise_and_hare.json")

    def test_character_and_scene_prompts_are_short_distinct_and_specific(self):
        positive = scene_prompt(self.story.scenes[0], self.bible)
        for text in ("Exactly two animals", "white fur", "pink inner ears", "blue vest", "green skin",
                     "visible rounded brown patterned shell", "sunny forest", "starting line", "unified cinematic frame"):
            self.assertIn(text, positive)
        self.assertLess(len(positive.split()), 60)
        for text in ("split panels", "character sheet", "triptych", "extra characters", "text"):
            self.assertIn(text, SCENE_NEGATIVE_PROMPT)
        for character in self.bible.characters.values():
            prompt = reference_prompt(character, self.bible)
            self.assertIn("Exactly one", prompt)
            self.assertIn("full-body", prompt)
            self.assertIn("three-quarter", prompt)
            self.assertNotIn("starting line", prompt)
            self.assertLess(len(prompt.split()), 60)
        self.assertIn("back shell clearly visible", reference_prompt(self.bible.characters["tortoise"], self.bible))
        self.assertIn("pedestal", REFERENCE_NEGATIVE_PROMPT)
        self.assertIn("multiple views", REFERENCE_NEGATIVE_PROMPT)

    def test_new_package_contains_only_challenge_and_no_historical_reference_pixels(self):
        result = prepare(self.root)
        with zipfile.ZipFile(result["package_path"]) as archive:
            request = json.loads(archive.read("request.json"))
            references = json.loads(archive.read("references.json"))
            self.assertEqual(references["references"], [])
            self.assertFalse(any(name.endswith(".png") for name in archive.namelist()))
            self.assertIn("src/image_conditioning.py", archive.namelist())
        self.assertEqual([s["id"] for s in request["scenes"]], ["challenge"])
        self.assertFalse(request["generation_policy"]["full_batch_enabled"])
        self.assertFalse(request["quality_review"]["approved"])
        self.assertEqual(request["prompt_revision"], PROMPT_REVISION)
        self.assertNotEqual(request["characters"][0]["negative_prompt"], request["scenes"][0]["negative_prompt"])
        # A repeat build of identical content is non-destructive and deterministic.
        self.assertEqual(prepare(self.root)["package_sha256"], result["package_sha256"])

    def test_new_references_must_be_reviewed_before_one_scene_and_full_is_blocked(self):
        artifacts = prepare(self.root)
        request_path = Path(artifacts["request_path"])
        result_dir = self.root / "result"
        model = SyntheticBackend()
        worker = BatchWorker(request_path, result_dir)
        data = worker.execute(lambda *args: model, mode="references", max_new_images=2)
        self.assertEqual(data["status"], "references_ready")
        self.assertEqual([kind for kind, _, _ in model.calls], ["character", "character"])
        worker = BatchWorker(request_path, result_dir)
        blocked = worker.preflight(mode="smoke", max_new_images=1, allow_unapproved_references=True)
        self.assertTrue(any("REFERENCES_REVIEWED" in b for b in blocked["blockers"]))
        data = worker.execute(lambda *args: model, mode="smoke", max_new_images=1,
                              allow_unapproved_references=True, references_reviewed=True)
        self.assertEqual([identity for kind, identity, _ in model.calls if kind == "scene"], ["challenge"])
        self.assertNotIn("smoke_quality_review", data)
        self.assertTrue(any("Full-batch generation is disabled" in b for b in worker.preflight(
            mode="full", max_new_images=0, quality_reviewed=True, allow_unapproved_references=True)["blockers"]))

    def test_historical_refs_are_preserved_and_not_silently_reused_for_new_spec(self):
        artifacts = prepare(self.root)
        path = self.root / "old.png"
        Image.new("RGB", (128, 128), "white").save(path)
        index = self.root / "references.json"
        write_json(index, {"references": [{"target_id": "hare", "file": path.name, "sha256": file_hash(path), "metadata": {}}]})
        original = path.read_bytes()
        worker = BatchWorker(artifacts["request_path"], self.root / "result", references=reference_sources(index))
        plan = worker.preflight(mode="references", max_new_images=2)
        self.assertEqual(plan["new_image_count"], 2)
        self.assertEqual(plan["reused_images"], [])
        self.assertTrue(any("Historical" in w for w in plan["warnings"]))
        self.assertEqual(path.read_bytes(), original)

    def test_masks_are_separate_stable_attention_regions_not_a_reference_collage(self):
        layout = reference_layout(["tortoise", "hare"])
        self.assertEqual(layout["reference_order"], ["hare", "tortoise"])
        masks, hashes = reference_masks(layout, (1024, 576))
        self.assertEqual([m.mode for m in masks], ["L", "L"])
        self.assertEqual(masks[0].getpixel((300, 300)), 255)
        self.assertEqual(masks[1].getpixel((300, 300)), 0)
        self.assertEqual(masks[1].getpixel((750, 350)), 255)
        self.assertEqual(masks[0].getpixel((750, 350)), 0)
        self.assertEqual([m.getpixel((500, 20)) for m in masks], [0, 0])
        self.assertEqual(reference_masks(reference_layout(["hare", "tortoise"]), (1024, 576))[1], hashes)

    def test_generation_identity_changes_for_prompt_settings_code_or_reference_binding(self):
        base = generation_identity("request_a", BatchSettings())
        self.assertNotEqual(base, generation_identity("request_b", BatchSettings()))
        self.assertNotEqual(base, generation_identity("request_a", BatchSettings(adapter_scale=.4)))
        self.assertNotEqual(base, generation_identity("request_a", BatchSettings(family="sd15")))
        with patch("src.image_batch.file_hash", return_value="changed_implementation"):
            self.assertNotEqual(base, generation_identity("request_a", BatchSettings()))
        # Reference content changes must not reuse an existing successful scene.
        request = self.root / "request.json"
        export_request(AssetStore(self.root / "store"), self.story, self.bible, request, ["challenge"])
        model = SyntheticBackend()
        worker = BatchWorker(request, self.root / "result")
        worker.execute(lambda *args: model, mode="smoke", allow_unapproved_references=True, max_new_images=3)
        old = next(a for a in worker.data["assets"] if a["kind"] == "character" and a["target_id"] == "hare")
        new_path = self.root / "different_hare.png"
        Image.new("RGB", (128, 128), "blue").save(new_path)
        replacement = {"target_id": "hare", "source_path": str(new_path), "sha256": file_hash(new_path), "metadata": {}}
        replacement_worker = BatchWorker(request, self.root / "result", references=[replacement])
        plan = replacement_worker.preflight(mode="smoke", max_new_images=0, allow_unapproved_references=True,
            reference_ids={"hare": replacement["sha256"]})
        self.assertTrue(any("Reference binding changed" in b for b in plan["blockers"]))
        self.assertTrue((self.root / "result" / old["file"]).exists())
        self.assertEqual(scene_seed("challenge"), 788870362)  # Existing stable-ID formula remains unchanged.


class BackendWiringTests(unittest.TestCase):
    def backend(self, token_limit=77):
        backend = ColabDiffusersBackend.__new__(ColabDiffusersBackend)
        backend.settings = BatchSettings()
        backend.provenance = {"model_id": "fixture", "model_revision": "v2", "adapter_id": "h94/IP-Adapter", "adapter_revision": "fixture"}
        backend.torch = MagicMock()
        backend.torch.cuda.max_memory_allocated.return_value = 1
        backend.pipe = MagicMock()
        tokenizer = MagicMock()
        tokenizer.model_max_length = token_limit
        tokenizer.side_effect = lambda text, truncation: {"input_ids": list(range(len(text.split()) + 2))}
        backend.pipe.tokenizer = backend.pipe.tokenizer_2 = tokenizer
        backend.pipe._execution_device = "cuda"
        backend.pipe.unet = SimpleNamespace(dtype="float16")
        backend.pipe.return_value = SimpleNamespace(images=[Image.new("RGB", (1024, 576))], nsfw_content_detected=None)
        backend._adapter = MagicMock()
        backend.adapter_loaded = False
        return backend

    def test_actual_backend_passes_nested_separate_images_masks_and_per_image_scales(self):
        backend = self.backend()
        processor = MagicMock()
        tensor = processor.preprocess.return_value.reshape.return_value.to.return_value
        module = ModuleType("diffusers.image_processor")
        module.IPAdapterMaskProcessor = lambda: processor
        refs = {"tortoise": Image.new("RGB", (768, 768), "green"), "hare": Image.new("RGB", (768, 768), "white")}
        job = {"kind": "scene", "prompt": "Exactly two animals on one sunny forest starting line.",
               "negative_prompt": SCENE_NEGATIVE_PROMPT, "seed": 7, "size": (1024, 576)}
        with patch.dict(sys.modules, {"diffusers": ModuleType("diffusers"), "diffusers.image_processor": module}):
            _, metadata = backend.generate(job, refs)
        call = backend.pipe.call_args.kwargs
        self.assertEqual(call["ip_adapter_image"], [[refs["hare"], refs["tortoise"]]])
        backend.pipe.set_ip_adapter_scale.assert_called_once_with([[.35, .35]])
        processor.preprocess.return_value.reshape.assert_called_once_with(1, 2, 576, 1024)
        self.assertEqual(call["cross_attention_kwargs"], {"ip_adapter_masks": [tensor]})
        self.assertEqual(metadata["conditioning"], CONDITIONING_REVISION)
        self.assertEqual(metadata["conditioning_layout"]["reference_order"], ["hare", "tortoise"])
        self.assertEqual(set(metadata["mask_sha256"]), {"hare", "tortoise"})

    def test_overlong_prompt_fails_before_adapter_or_inference_instead_of_truncating(self):
        backend = self.backend(token_limit=3)
        with self.assertRaisesRegex(ValueError, "would be truncated"):
            backend.generate({"kind": "scene", "prompt": "too many important subject tokens here",
                              "negative_prompt": "text", "seed": 1, "size": (1024, 576)}, {"hare": Image.new("RGB", (64, 64))})
        backend._adapter.assert_not_called()
        backend.pipe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
