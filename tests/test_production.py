"""Real lightweight FFmpeg integration; voices/images here are synthetic fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.asset_store import AssetStore
from src.media import find_tool, validate_video
from src.narration import NarrationResult, write_silence
from src.production import ROOT, produce
from src.production_providers import FFmpegAnimation
from src.production_state import file_hash


class FixtureVoice:
    cache_key = "test_fixture_voice_v1"

    def __init__(self):
        self.calls = []

    def synthesize(self, scene, directory, settings):
        self.calls.append(scene.id)
        path = directory / f"{scene.id}.wav"
        write_silence(path, .25)
        return NarrationResult(path, .25, "synthetic_test_fixture")


class FailOnceAnimation:
    def __init__(self, real):
        self.real, self.cache_key, self.calls = real, real.cache_key, 0

    def animate(self, *args):
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("Injected clip failure")
        return self.real.animate(*args)


class ProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.ffmpeg, cls.ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="production tests ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "story.json"
        self.script_data = json.loads((ROOT / "scripts/tortoise_and_hare.json").read_text())
        self.script_data["scenes"] = self.script_data["scenes"][:2]
        for scene in self.script_data["scenes"]:
            scene["duration"] = .4
        self.script.write_text(json.dumps(self.script_data))
        self.pipeline = self.root / "pipeline.json"
        config = json.loads((ROOT / "configs/pipeline.json").read_text())
        config.update(fps=12, tail_seconds=0)
        self.pipeline.write_text(json.dumps(config))
        self.settings = self.root / "settings.json"
        self.settings.write_text(json.dumps({"vertical_scene_ids": ["challenge"]}))
        self.store = AssetStore(self.root / "library")
        self.voice = FixtureVoice()
        self.out = self.root / "production"

    def run_production(self, **options):
        return produce(self.script, self.out, settings_path=self.settings, pipeline_path=self.pipeline,
                       asset_dir=self.store.root, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe,
                       voice=self.voice, **options)

    def candidate(self, kind, target):
        path = self.root / f"{kind}_{target}.png"
        Image.new("RGB", (800, 400), "#5971aa").save(path)
        return self.store.import_image(path, kind, target, {"model_id": "synthetic_test_fixture"})

    def test_complete_execution_unapproved_fallback_and_output_validation(self):
        candidate = self.candidate("scene", "challenge")
        reference = self.candidate("character", "hare")
        result = self.run_production()
        self.assertEqual(result["status"], "completed")
        self.assertEqual([s["id"] for s in result["scenes"]], ["challenge", "racing_ahead"])
        self.assertTrue(all(s["status"] == "completed" for s in result["scenes"]))
        self.assertEqual(result["scenes"][0]["images"]["landscape"]["provider"], "pillow")
        self.assertIn("fallback_reason", result["scenes"][0]["images"]["landscape"])
        self.assertIn(candidate, result["asset_summary"]["awaiting_approval"])
        self.assertIn(reference, result["asset_summary"]["awaiting_approval"])
        self.assertEqual(self.store.read()["approvals"], {"scene": {}, "character": {}})
        request = json.loads(Path(result["ai_generation_request"]["path"]).read_text())
        self.assertEqual([s["id"] for s in request["scenes"]], ["challenge", "racing_ahead"])
        self.assertTrue(all("approved_reference_png_base64" not in c for c in request["characters"]))
        self.assertFalse(request["production_notes"]["generation_started"])
        for name, dimensions in (("landscape", (1280, 720)), ("vertical", (720, 1280))):
            output = result["outputs"][name]
            info = validate_video(self.ffprobe, Path(output["path"]), dimensions)
            self.assertEqual(info["audio_codec"], "aac")
            self.assertTrue(output["full_decode_passed"])
        with self.assertRaises(ValueError):
            validate_video(self.ffprobe, Path(result["outputs"]["landscape"]["path"]), (1, 1))

    def test_idempotent_rerun_reuses_every_stage_without_touching_outputs(self):
        first = self.run_production()
        originals = {o["path"]: (file_hash(o["path"]), Path(o["path"]).stat().st_mtime_ns) for o in first["outputs"].values()}
        second = self.run_production()
        self.assertEqual(self.voice.calls, ["challenge", "racing_ahead"])
        self.assertEqual(second["runs"][-1]["executed_stages"], [])
        self.assertEqual(len(second["runs"][-1]["reused_stages"]), 11)
        for path, expected in originals.items():
            self.assertEqual((file_hash(path), Path(path).stat().st_mtime_ns), expected)

    def test_failed_scene_is_reported_and_resume_keeps_successful_work(self):
        animation = FailOnceAnimation(FFmpegAnimation(self.ffmpeg))
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            self.run_production(animation=animation)
        failed = json.loads((self.out / "production_manifest.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure"]["stage"], "clip:racing_ahead:landscape")
        self.assertEqual(failed["scenes"][0]["status"], "completed")
        self.assertEqual(failed["scenes"][1]["status"], "failed")
        resumed = self.run_production(animation=animation)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(len(self.voice.calls), 2)
        self.assertIn("clip:challenge:landscape", resumed["runs"][-1]["reused_stages"])
        self.assertIn("clip:racing_ahead:landscape", resumed["runs"][-1]["executed_stages"])
        self.assertEqual(resumed["runs"][0]["status"], "failed")

    def test_changed_scene_invalidates_only_affected_work_and_keeps_old_media(self):
        first = self.run_production()
        old_path = first["outputs"]["landscape"]["path"]
        old_hash = file_hash(old_path)
        self.script_data["scenes"][1]["narration"] = "The friends begin their race."
        self.script.write_text(json.dumps(self.script_data))
        second = self.run_production()
        self.assertEqual(self.voice.calls, ["challenge", "racing_ahead", "racing_ahead"])
        run = second["runs"][-1]
        self.assertIn("voice:challenge", run["reused_stages"])
        self.assertIn("voice:racing_ahead", run["executed_stages"])
        self.assertIn("final:vertical", run["reused_stages"])
        self.assertIn("final:landscape", run["executed_stages"])
        self.assertEqual(file_hash(old_path), old_hash)

    def test_corrupted_final_is_rebuilt_without_overwriting_previous_attempt(self):
        first = self.run_production()
        old_path = Path(first["outputs"]["landscape"]["path"])
        old_path.write_bytes(b"damaged final file")
        second = self.run_production()
        self.assertEqual(second["runs"][-1]["executed_stages"], ["final:landscape"])
        self.assertNotEqual(second["outputs"]["landscape"]["path"], str(old_path))
        self.assertEqual(old_path.read_bytes(), b"damaged final file")
        self.assertEqual(len(self.voice.calls), 2)

    def test_advisory_duration_warns_without_stretching_short_story(self):
        self.settings.write_text(json.dumps({"target_duration": 300, "vertical_scene_ids": ["challenge"]}))
        result = self.run_production()
        self.assertLess(result["outputs"]["landscape"]["duration"], 2)
        self.assertTrue(any("300.00s" in warning for warning in result["warnings"]))

    def test_approved_selection_integrity_and_request_locks(self):
        identity = self.candidate("scene", "challenge")
        self.store.approve(identity)  # Isolated synthetic fixture only, never the real library.
        first = self.run_production()
        self.assertEqual(first["scenes"][0]["images"]["landscape"]["provider"], "approved_file")
        self.assertEqual(first["asset_summary"]["needs_ai_generation"], ["racing_ahead"])
        path = self.store.approved("scene", "challenge")[1]
        path.write_bytes(b"tampered approved asset")
        second = self.run_production()
        self.assertEqual(second["scenes"][0]["images"]["landscape"]["provider"], "pillow")
        request = json.loads(Path(second["ai_generation_request"]["path"]).read_text())
        self.assertEqual(request["locked_scene_ids"], ["challenge"])
        self.assertEqual(self.store.read()["approvals"]["scene"]["challenge"], identity)
        self.assertEqual(len(self.voice.calls), 2)

    def test_planning_failure_is_checkpointed_and_foreign_outputs_are_protected(self):
        self.settings.write_text(json.dumps({"language": "fr"}))
        with self.assertRaisesRegex(ValueError, "English"):
            self.run_production()
        result = json.loads((self.out / "production_manifest.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["stage"], "planning")
        foreign = self.root / "old_outputs"
        foreign.mkdir()
        old = foreign / "landscape.mp4"
        old.write_bytes(b"preserve this existing output")
        with self.assertRaisesRegex(ValueError, "NEW output"):
            produce(self.script, foreign)
        self.assertEqual(old.read_bytes(), b"preserve this existing output")
        self.assertFalse((foreign / "production_manifest.json").exists())

    def test_refreshed_narration_remains_selected_on_next_normal_resume(self):
        self.run_production()
        refreshed = self.run_production(refresh_narration=True)
        next_run = self.run_production()
        self.assertEqual(len(self.voice.calls), 4)
        self.assertEqual(next_run["runs"][-1]["executed_stages"], [])
        self.assertEqual(next_run["scenes"][0]["narration"]["path"], refreshed["scenes"][0]["narration"]["path"])

    def test_vertical_only_does_not_render_or_request_unused_scenes(self):
        self.settings.write_text(json.dumps({"formats": ["vertical"], "vertical_scene_ids": ["challenge"]}))
        result = self.run_production()
        self.assertEqual(set(result["outputs"]), {"vertical"})
        self.assertEqual(self.voice.calls, ["challenge"])
        self.assertEqual(result["scenes"][1]["status"], "not_requested")
        self.assertEqual(result["asset_summary"]["placeholder_scene_ids"], ["challenge"])
        self.assertEqual(result["ai_generation_request"]["scene_ids"], ["challenge"])

    def test_failure_after_success_does_not_report_stale_current_outputs(self):
        first = self.run_production()
        old_path = first["outputs"]["landscape"]["path"]
        old_hash = file_hash(old_path)
        self.settings.write_text(json.dumps({"animation": "unimplemented_ai_video"}))
        with self.assertRaisesRegex(ValueError, "zoom_pan"):
            self.run_production()
        failed = json.loads((self.out / "production_manifest.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["outputs"], {})
        self.assertNotIn("quality", failed)
        self.assertEqual(file_hash(old_path), old_hash)
        self.assertEqual(failed["runs"][0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
