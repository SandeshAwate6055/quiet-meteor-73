import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.asset_store import AssetStore
from src.assets import export_request, import_batch, review
from src.characters import load_bible, scene_prompt
from src.image_providers import FileImageProvider
from src.main import ROOT, build
from src.media import find_tool
from src.schema import load_story


class CharacterTests(unittest.TestCase):
    def test_bible_and_prompt_preserve_identity(self):
        bible = load_bible(ROOT / "configs/character_bible.json")
        self.assertIn("white fur", bible.characters["hare"].appearance)
        self.assertIn("blue vest", bible.characters["hare"].appearance)
        self.assertIn("brown patterned shell", bible.characters["tortoise"].appearance)
        scene = load_story(ROOT / "scripts/tortoise_and_hare.json").scenes[0]
        self.assertIn("blue vest", scene_prompt(scene, bible))
        self.assertTrue(bible.characters["hare"].reference_image_paths)

    def test_reject_duplicate_identity_and_escaping_reference(self):
        data = json.loads((ROOT / "configs/character_bible.json").read_text())
        for duplicate in (True, False):
            changed = copy.deepcopy(data)
            if duplicate:
                changed["characters"][1]["id"] = "hare"
            else:
                changed["characters"][0]["reference_image_paths"] = ["../escape.png"]
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bible.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_bible(path)


class NotebookTests(unittest.TestCase):
    def test_notebook_syntax_unexecuted_and_small_batch_defaults(self):
        notebook = json.loads((ROOT / "notebooks/phase2_image_generation.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        assignments = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            compile(source, "colab_cell", "exec")
            for statement in ast.parse(source).body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    try:
                        assignments[statement.targets[0].id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        pass
        self.assertEqual(assignments["SELECT_SCENES"], ["challenge"])
        self.assertEqual(assignments["MAX_NEW_IMAGES"], 3)
        self.assertFalse(assignments["RUN_GENERATION"])
        self.assertFalse(assignments["INSTALL_COLAB_DEPS"])
        self.assertFalse(assignments["REFERENCES_OK"])


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = AssetStore(self.root / "library")
        self.story = load_story(ROOT / "scripts/tortoise_and_hare.json")
        self.bible = load_bible(ROOT / "configs/character_bible.json")
        self.scene = self.story.scenes[0]

    def source(self, name="source.png", size=(800, 400), color="red"):
        path = self.root / name
        Image.new("RGB", size, color).save(path)
        return path

    def render(self, size=(1280, 720)):
        path = self.root / "render.png"
        record = FileImageProvider(self.store).render(self.scene, self.story.title, 1, 5, path, size)
        with Image.open(path) as image:
            self.assertEqual(image.size, size)
        return record

    def test_missing_and_unapproved_fall_back(self):
        self.assertEqual(self.render()["provider"], "pillow")
        self.store.import_image(self.source(), "scene", self.scene.id)
        self.assertEqual(self.render()["provider"], "pillow")

    def test_resolution_adaptation_and_integrity_fallback(self):
        identity = self.store.import_image(self.source(), "scene", self.scene.id)
        self.store.approve(identity)
        for size in ((1280, 720), (720, 1280)):
            record = self.render(size)
            self.assertEqual(record["source_resolution"], [800, 400])
            self.assertEqual(record["provider"], "approved_file")
        path = self.store.approved("scene", self.scene.id)[1]
        path.write_bytes(b"broken")
        self.assertIn("modified", self.render()["fallback_reason"])
        path.unlink()
        self.assertEqual(self.render()["provider"], "pillow")

    def test_deterministic_selection_lock_and_explicit_unlock(self):
        first = self.store.import_image(self.source(), "scene", self.scene.id)
        self.store.approve(first)
        original = self.store.approved("scene", self.scene.id)[1].read_bytes()
        second = self.store.import_image(self.source("second.png", color="blue"), "scene", self.scene.id)
        self.assertEqual(self.store.import_image(self.root / "source.png", "scene", self.scene.id), first)
        self.assertEqual(self.render()["asset_id"], first)
        with self.assertRaisesRegex(ValueError, "locked"):
            self.store.approve(second)
        self.assertEqual(self.store.approved("scene", self.scene.id)[1].read_bytes(), original)
        self.store.unlock("scene", self.scene.id, "Reviewed replacement")
        self.store.approve(second)
        self.assertEqual(self.render()["asset_id"], second)

    def test_reject_tiny_images(self):
        with self.assertRaises(ValueError):
            self.store.import_image(self.source(size=(20, 20)), "scene", self.scene.id)

    def test_batch_import_review_and_export_round_trip(self):
        source = self.source()
        manifest = self.root / "manifest.json"
        data = {"version": 1, "assets": [{"kind": "character", "target_id": "hare", "file": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "metadata": {"seed": 42, "model_id": "synthetic-test-fixture"}}]}
        manifest.write_text(json.dumps(data))
        identities = import_batch(self.store, manifest, self.story, self.bible)
        catalog = self.store.read()
        imported = catalog["assets"][identities[0]]
        self.assertEqual(imported["metadata"]["seed"], 42)
        self.assertEqual(imported["metadata"]["source_batch"]["manifest_path"], str(manifest.resolve()))
        self.assertEqual(imported["metadata"]["source_batch"]["source_sha256"], data["assets"][0]["sha256"])
        self.assertEqual(import_batch(self.store, manifest, self.story, self.bible), identities)
        self.assertEqual(self.store.read()["assets"], catalog["assets"])
        self.assertFalse(self.store.read()["approvals"]["character"])
        self.store.approve(identities[0])
        destination = self.root / "request.json"
        export_request(self.store, self.story, self.bible, destination)
        request = json.loads(destination.read_text())
        self.assertEqual(len(request["scenes"]), 4)
        self.assertIn("approved_reference_png_base64", request["characters"][0])
        self.assertTrue(Path(review(self.store, self.root / "review")["pages"][0]).exists())
        data["assets"][0]["file"] = "../escape.png"
        manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            import_batch(self.store, manifest, self.story, self.bible)

    def test_full_pipeline_mixed_provider_regression(self):
        try:
            find_tool("ffmpeg")
            find_tool("ffprobe")
        except RuntimeError as exc:
            self.skipTest(str(exc))
        # Synthetic fixture verifies file integration, not AI generation; short scenes bound test cost.
        source = self.source(color="#528b58")
        identity = self.store.import_image(source, "scene", "challenge", {"model_id": "synthetic-test-fixture"})
        self.store.approve(identity)
        script = json.loads((ROOT / "scripts/tortoise_and_hare.json").read_text())
        script["scenes"] = script["scenes"][:2]
        for row in script["scenes"]:
            row["duration"] = .6
        script_path = self.root / "short.json"
        script_path.write_text(json.dumps(script))
        args = argparse.Namespace(script=script_path, config=ROOT / "configs/pipeline.json", ffmpeg=None, ffprobe=None,
            character_bible=ROOT / "configs/character_bible.json", image_provider="auto", asset_dir=self.store.root,
            vertical_scenes=["challenge"], output_dir=self.root / "outputs", silent=True)
        result = build(args)
        self.assertEqual(result["scenes"][0]["images"]["landscape"]["provider"], "approved_file")
        self.assertEqual(result["scenes"][1]["images"]["landscape"]["provider"], "pillow")
        self.assertEqual(result["outputs"]["landscape"]["width"], 1280)
        self.assertEqual(result["outputs"]["vertical"]["height"], 1280)
        self.assertEqual(result["outputs"]["landscape"]["audio_codec"], "aac")


if __name__ == "__main__":
    unittest.main()
