"""Tests for Phase 6 long-form 180-240s episode production, reuse, Shorts, and one-click batch."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from src.asset_store import AssetStore
from src.characters import load_bible
from src.image_batch import scene_seed
from src.media import find_tool, motion_filter
from src.modern_images import FluxKleinBackend, validate_profile
from src.narration import NarrationResult, write_silence
from src.one_click import (
    DEFAULT_EPISODE_PACKAGE_NAME,
    ROOT,
    extract_and_verify_bundle,
    load_challenge_profile,
    prepare_one_click_episode_handoff,
    run_one_click_batch,
)
from src.production import produce
from src.production_plan import load_settings
from src.schema import MOTIONS, load_story
from src.shorts import plan_shorts, render_short
from test_one_click import SyntheticKleinBackend


class EpisodeProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.story_path = ROOT / "scripts/episode_01_golden_acorn.json"
        self.story = load_story(self.story_path)
        self.bible = load_bible(ROOT / "configs/character_bible.json")

    def _seeded_store(self):
        store = AssetStore(self.root / "library")
        base_profile, _, _ = load_challenge_profile()
        for identity, color in (("hare", "white"), ("tortoise", "green")):
            p = self.root / f"ref_{identity}.png"
            Image.new("RGB", (1024, 1024), color).save(p)
            aid = store.import_image(p, "character", identity, {
                "model_id": base_profile["model_id"],
                "model_revision": base_profile["model_revision"],
                "model_license": base_profile["model_license"],
                "reference_spec_revision": "reference_flux_klein_v4",
            })
            store.approve(aid)
        for sid in ("challenge", "racing_ahead", "nap", "steady_steps", "finish"):
            p = self.root / f"scene_{sid}.png"
            Image.new("RGB", (1024, 576), "#4caf50").save(p)
            aid = store.import_image(p, "scene", sid, {
                "model_id": base_profile["model_id"],
                "model_revision": base_profile["model_revision"],
                "model_license": base_profile["model_license"],
            })
            store.approve(aid)
        return store

    def test_long_form_story_storyboard_completeness_and_duration_target(self):
        self.assertEqual(self.story.title, "The Golden Acorn Race")
        self.assertTrue(24 <= len(self.story.scenes) <= 30)
        self.assertEqual(self.story.meta["target_duration"], 210)
        words = sum(len(s.narration.split()) for s in self.story.scenes)
        pauses = sum(s.pause_after for s in self.story.scenes)
        est_total = words / 2.5 + pauses
        self.assertTrue(180 <= est_total <= 240, f"Estimated duration {est_total:.1f}s outside 180-240s")

        reused = [s for s in self.story.scenes if s.reuse_asset]
        self.assertEqual(
            {s.reuse_asset for s in reused},
            {"challenge", "racing_ahead", "nap", "steady_steps", "finish"},
        )
        for sc in self.story.scenes:
            self.assertTrue(sc.camera)
            self.assertIn(sc.motion, MOTIONS)
            self.assertTrue(sc.continuity)
            self.assertTrue(0.0 <= sc.focus <= 1.0)
            self.assertTrue(sc.characters)

        shorts = plan_shorts(self.story)
        self.assertEqual(len(shorts), 2)
        self.assertTrue(all(sh["shot_count"] >= 6 for sh in shorts))

    def test_episode_handoff_generates_missing_only_and_supports_partial_resume(self):
        store = self._seeded_store()
        handoff = prepare_one_click_episode_handoff(
            self.story_path, self.root / "ep_handoff", asset_dir=store.root
        )
        self.assertEqual(len(handoff["reused_shot_ids"]), 5)
        self.assertEqual(len(handoff["scene_ids"]), len(self.story.scenes) - 5)
        self.assertEqual(
            set(handoff["excluded_approved_scene_ids"]),
            {"challenge", "racing_ahead", "nap", "steady_steps", "finish"},
        )
        for approved_id in ("challenge", "racing_ahead", "nap", "steady_steps", "finish"):
            self.assertNotIn(approved_id, handoff["scene_ids"])

        bundle = extract_and_verify_bundle(
            handoff["package_path"],
            self.root / "ep_bundles",
            expected_package_sha256=handoff["package_sha256"],
        )
        validate_profile(bundle["profile"])
        self.assertEqual(bundle["profile"]["mode"], "episode_scenes")
        self.assertEqual(bundle["profile"]["max_new_images"], 24)

        # Simulate failure on shot 3 ('crossroads_greeting'), partial export, then resume:
        backend = SyntheticKleinBackend(bundle["profile"], fail_id="crossroads_greeting")
        loads = 0

        def factory(settings, previous):
            nonlocal loads
            loads += 1
            return backend

        work_root = self.root / "ep_work"
        partial = run_one_click_batch(
            bundle["bundle_dir"], work_root, backend_factory=factory, allow_partial=True
        )
        self.assertEqual(partial["outcome"]["status"], "failed")
        self.assertEqual(partial["validation"]["completed_scene_ids"], ["hook_blur", "tortoise_morning"])
        self.assertEqual(partial["validation"]["failed_scene_ids"], ["crossroads_greeting"])
        self.assertTrue(partial["archive_path"].is_file())

        backend.fail_id = None
        resumed = run_one_click_batch(bundle["bundle_dir"], work_root, backend_factory=factory)
        self.assertEqual(resumed["outcome"]["status"], "complete")
        self.assertEqual(len(resumed["validation"]["completed_scene_ids"]), 24)
        self.assertTrue(all(c[0] == "scene" for c in backend.calls))
        # hook_blur and tortoise_morning were NOT regenerated on resume:
        self.assertEqual(sum(1 for _, sid, _ in backend.calls if sid == "hook_blur"), 1)
        self.assertEqual(sum(1 for _, sid, _ in backend.calls if sid == "tortoise_morning"), 1)

    def test_episode_notebook_is_lightweight_and_simulates_end_to_end(self):
        nb_path = ROOT / "notebooks/one_click_episode_scenes.ipynb"
        self.assertTrue(nb_path.is_file())
        self.assertLess(nb_path.stat().st_size, 32 * 1024)
        raw = nb_path.read_text(encoding="utf-8")
        self.assertNotIn("EMBEDDED_PACKAGE_B64", raw)
        self.assertRegex(
            raw,
            r"https://raw\.githubusercontent\.com/SandeshAwate6055/quiet-meteor-73/[0-9a-f]{40}/examples/",
        )
        nb = json.loads(raw)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 5)
        for cell in code_cells:
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])

        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda idx: (8, 9),
            get_device_properties=lambda idx: SimpleNamespace(total_memory=int(22.5 * 2**30)),
            get_device_name=lambda idx: "NVIDIA L4",
        )
        downloads, displayed = [], []
        fake_colab = ModuleType("google.colab")
        fake_colab.files = SimpleNamespace(download=lambda path: downloads.append(path))
        fake_colab.drive = SimpleNamespace(mount=lambda path: (_ for _ in ()).throw(AssertionError("No Drive")))
        fake_google = ModuleType("google")
        fake_google.colab = fake_colab
        fake_display = ModuleType("IPython.display")
        fake_display.display = lambda obj: displayed.append(obj)

        work_dir = self.root / "ep_nb_sim"
        env = {}
        c1 = "".join(code_cells[0]["source"]).replace('PERSISTENT_DIR = ""', f'PERSISTENT_DIR = {json.dumps(str(work_dir))}')
        c2 = "".join(code_cells[1]["source"]).replace(
            'meminfo = Path("/proc/meminfo")',
            'meminfo = WORK_ROOT / "fake_meminfo"\nmeminfo.write_text("MemTotal: 52428800 kB\\n")',
        )
        c3 = "".join(code_cells[2]["source"])
        c4 = "".join(code_cells[3]["source"])
        c5 = "".join(code_cells[4]["source"])
        req_pins = {
            "diffusers": "0.37.1", "transformers": "4.57.6", "accelerate": "1.12.0",
            "huggingface-hub": "0.36.2", "safetensors": "0.7.0", "Pillow": "12.1.1",
        }
        pkg_path = ROOT / f"examples/{DEFAULT_EPISODE_PACKAGE_NAME}.zip"
        pkg_bytes = pkg_path.read_bytes()
        bundle_tmp = extract_and_verify_bundle(pkg_path, self.root / "tmp_ep_profile")
        synthetic_backend = SyntheticKleinBackend(bundle_tmp["profile"])

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return pkg_bytes

        prev_path = list(sys.path)
        prev_mods = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
        try:
            with patch.dict(sys.modules, {
                "torch": fake_torch, "google": fake_google, "google.colab": fake_colab, "IPython.display": fake_display,
            }), patch("urllib.request.urlopen", return_value=FakeResponse()), \
                    patch("importlib.metadata.version", side_effect=lambda pkg: req_pins[pkg]):
                exec(compile(c1, "cell1", "exec"), env)
                exec(compile(c2, "cell2", "exec"), env)
                exec(compile(c3, "cell3", "exec"), env)
                exec(compile(c4, "cell4", "exec"), env)
                with patch("src.modern_images.FluxKleinBackend", return_value=synthetic_backend):
                    exec(compile(c5, "cell5", "exec"), env)
        finally:
            sys.path[:] = prev_path
            for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
                sys.modules.pop(k, None)
            sys.modules.update(prev_mods)

        self.assertEqual(len(downloads), 1)
        self.assertTrue(Path(downloads[0]).is_file())
        self.assertEqual(len(synthetic_backend.calls), 24)

    def test_music_licensing_guard_motion_presets_and_reused_scene_rendering(self):
        # Unlicensed music config is rejected:
        bad_settings = self.root / "bad_music.json"
        bad_settings.write_text(json.dumps({"music": {"path": "x.wav"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unlicensed audio is refused"):
            load_settings(bad_settings)

        for preset in MOTIONS:
            self.assertIn("zoompan=", motion_filter(preset, 96, 1280, 720, 24))

        try:
            ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
        except RuntimeError as exc:
            self.skipTest(str(exc))

        store = self._seeded_store()
        # Mini 2-shot story testing reuse_asset, pause_after, motion, duration_range, and Shorts:
        mini_story = self.root / "mini_episode.json"
        mini_story.write_text(json.dumps({
            "title": "Mini Reuse Test",
            "target_duration": 2.0,
            "scenes": [
                {"id": "opening_line", "narration": "Line one.", "duration": 0.4, "pause_after": 0.25,
                 "visual_description": "Start line.", "characters": ["hare", "tortoise"],
                 "reuse_asset": "challenge", "motion": "hold", "camera": "wide"},
                {"id": "middle_nap", "narration": "Line two.", "duration": 0.4, "pause_after": 0.25,
                 "visual_description": "Nap shot.", "characters": ["hare"],
                 "reuse_asset": "nap", "motion": "pan_right", "camera": "medium"},
            ],
            "shorts": [{"id": "short_mini", "title": "Mini Short", "shots": ["opening_line", "middle_nap"]}],
        }), encoding="utf-8")
        pipeline_path = self.root / "pipe.json"
        pipeline_cfg = json.loads((ROOT / "configs/pipeline.json").read_text(encoding="utf-8"))
        pipeline_cfg.update(fps=12, tail_seconds=0, vertical_scene_ids=["opening_line", "middle_nap"])
        pipeline_path.write_text(json.dumps(pipeline_cfg), encoding="utf-8")

        settings_path = self.root / "prod_settings.json"
        settings_path.write_text(json.dumps({"duration_range": [1.0, 5.0]}), encoding="utf-8")

        class Voice:
            cache_key = "mini_voice_v1"
            def synthesize(self, scene, directory, settings):
                p = directory / f"{scene.id}.wav"
                write_silence(p, 0.35)
                return NarrationResult(p, 0.35, "synthetic_voice")

        out_dir = self.root / "mini_prod"
        result = produce(mini_story, out_dir, settings_path=settings_path, pipeline_path=pipeline_path,
                         asset_dir=store.root, ffmpeg=ffmpeg, ffprobe=ffprobe, voice=Voice())
        self.assertTrue(result["quality"]["publish_ready"])
        self.assertEqual(result["asset_summary"]["needs_ai_generation"], [])
        self.assertTrue(all(s["images"]["landscape"]["provider"] == "approved_file" for s in result["scenes"]))
        self.assertGreaterEqual(result["outputs"]["landscape"]["duration"], 1.2)

        # Render standalone Short from the same assets:
        short_res = render_short(mini_story, "short_mini", self.root / "shorts_out",
                                 asset_dir=store.root, pipeline_path=pipeline_path,
                                 ffmpeg=ffmpeg, ffprobe=ffprobe, voice=Voice())
        self.assertEqual(short_res["output"]["width"], 720)
        self.assertEqual(short_res["output"]["height"], 1280)
        self.assertTrue(Path(short_res["output"]["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
