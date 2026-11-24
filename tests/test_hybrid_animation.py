import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from PIL import Image

from src import media
from src.asset_store import AssetStore
from src.assets import import_batch, review
from src.characters import load_bible
from src.image_conditioning import reference_layout
from src.modern_video import WanI2VBackend, check_i2v_hardware, validate_i2v_profile
from src.one_click import extract_and_verify_bundle, run_one_click_batch
from src.one_click_video import extract_and_verify_i2v_bundle, run_one_click_i2v_batch
from src.production_providers import HybridAnimation
from src.schema import load_story

ROOT = Path(__file__).resolve().parents[1]


def _make_dummy_mp4(path: Path, ffmpeg: str, size=(320, 180), duration=1.0, color="royalblue"):
    media.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s={size[0]}x{size[1]}:r=16",
        "-t", f"{duration:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ], 30)


def _make_dummy_wav(path: Path, ffmpeg: str, duration=1.2):
    media.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
        "-t", f"{duration:.2f}", path,
    ], 30)


class HybridAnimationTests(unittest.TestCase):
    def test_library_has_all_29_approved_episode_scenes(self):
        store = AssetStore(ROOT / "assets/library")
        catalog = store.read()
        self.assertEqual(len(catalog["approvals"]["character"]), 2)
        self.assertEqual(len(catalog["approvals"]["scene"]), 29)
        story = load_story(ROOT / "scripts/episode_01_golden_acorn.json")
        for sc in story.scenes:
            self.assertIsNotNone(store.approved("scene", sc.reuse_asset or sc.id))

    def test_corrections_notebook_is_lightweight_and_targets_3_scenes(self):
        nb_path = ROOT / "notebooks/one_click_episode_corrections.ipynb"
        raw = nb_path.read_text(encoding="utf-8")
        self.assertLess(nb_path.stat().st_size, 32 * 1024)
        self.assertNotIn("EMBEDDED_PACKAGE_B64", raw)
        nb = json.loads(raw)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])

        req = json.loads((ROOT / "examples/episode_01_corrections_v4_one_click.json").read_text(encoding="utf-8"))
        self.assertEqual([s["id"] for s in req["scenes"]], ["golden_acorn", "tortoise_stuck", "crossing_finish"])
        self.assertEqual(req["generation_config"]["base_seed"], 20261127)

        pkg = ROOT / "examples/episode_01_corrections_v4_one_click.zip"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            info = extract_and_verify_bundle(pkg, work / "bundles")

            class FakeBackend:
                def __init__(self, settings, profile):
                    self.settings = settings
                    self.calls = 0
                    self.provenance = {
                        "model_id": profile["model_id"],
                        "model_revision": profile["model_revision"],
                        "model_license": profile["model_license"],
                    }

                def generate(self, job, refs):
                    self.calls += 1
                    return Image.new("RGB", (1024, 576), "#4488aa"), {
                        "seed": job["seed"],
                        "effective_prompt": job["prompt"],
                        "model_id": info["profile"]["model_id"],
                        "model_revision": info["profile"]["model_revision"],
                        "model_license": info["profile"]["model_license"],
                        "conditioning_applied": bool(refs),
                        "conditioning": job["conditioning"],
                        "conditioning_layout": reference_layout(refs),
                    }

            out = run_one_click_batch(
                info["bundle_dir"],
                work,
                backend_factory=lambda s, p: FakeBackend(s, info["profile"]),
                cache_dir=work / "cache",
            )
            self.assertEqual(out["outcome"]["status"], "complete")
            self.assertEqual(len(out["outcome"]["assets"]), 5)  # 2 reused chars + 3 generated scenes
            self.assertEqual(len(out["validation"]["completed_scene_ids"]), 3)
            self.assertTrue(out["archive_path"].is_file())

    def test_asset_store_clip_lifecycle_and_import_batch(self):
        ffmpeg = media.find_tool("ffmpeg")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AssetStore(root / "lib")
            clip_src = root / "hare_sprints.mp4"
            _make_dummy_mp4(clip_src, ffmpeg)
            sha = hashlib.sha256(clip_src.read_bytes()).hexdigest()

            manifest = root / "batch" / "manifest.json"
            (root / "batch" / "clips").mkdir(parents=True)
            shutil.copy2(clip_src, root / "batch" / "clips" / "hare_sprints.mp4")
            manifest.write_text(json.dumps({
                "version": 1,
                "batch_id": "test_clip_batch",
                "assets": [{
                    "kind": "clip",
                    "target_id": "hare_sprints",
                    "file": "clips/hare_sprints.mp4",
                    "sha256": sha,
                    "metadata": {"seed": 123},
                }],
            }), encoding="utf-8")

            story = load_story(ROOT / "scripts/episode_01_golden_acorn.json")
            bible = load_bible(ROOT / "configs/character_bible.json")
            imported = import_batch(store, manifest, story, bible)
            self.assertEqual(len(imported), 1)
            asset_id = imported[0]
            self.assertTrue(asset_id.startswith("clip:hare_sprints:"))

            store.approve(asset_id)
            rec, approved_path = store.approved("clip", "hare_sprints")
            self.assertEqual(rec["asset_id"], asset_id)
            self.assertTrue(approved_path.is_file())

            rev = review(store, root / "review")
            self.assertEqual(len(rev["pages"]), 1)
            self.assertTrue(rev["assets"][0]["approved"])

            store.unlock("clip", "hare_sprints", "Testing unlock")
            self.assertIsNone(store.approved("clip", "hare_sprints"))

    def test_hybrid_animation_uses_approved_clip_and_fallback(self):
        ffmpeg = media.find_tool("ffmpeg")
        ffprobe = media.find_tool("ffprobe")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AssetStore(root / "lib")
            img_path = root / "scene.png"
            Image.new("RGB", (1024, 576), "#226644").save(img_path, "PNG")
            wav_path = root / "audio.wav"
            _make_dummy_wav(wav_path, ffmpeg, duration=1.2)
            clip_src = root / "hare_sprints.mp4"
            _make_dummy_mp4(clip_src, ffmpeg, size=(832, 480), duration=0.8)
            asset_id = store.import_clip(clip_src, "hare_sprints", {"seed": 42})

            hybrid = HybridAnimation(ffmpeg, store)
            settings = {"fps": 24, "threads": 2}
            # Before approval: camera-only fallback
            self.assertEqual(hybrid.clip_cache_key("hare_sprints"), "camera_only")
            fb_out = root / "fallback.mp4"
            dur_fb = hybrid.animate(img_path, wav_path, fb_out, 1.2, (1280, 720), settings, "push_in", scene_id="hare_sprints")
            media.validate_video(ffprobe, fb_out, (1280, 720), dur_fb)

            # After approval: uses approved I2V clip for both landscape and vertical without altering other shots
            store.approve(asset_id)
            self.assertNotEqual(hybrid.clip_cache_key("hare_sprints"), "camera_only")
            self.assertEqual(hybrid.clip_cache_key("challenge"), "camera_only")

            for name, size in (("landscape", (1280, 720)), ("vertical", (720, 1280))):
                out_mp4 = root / f"hybrid_{name}.mp4"
                dur = hybrid.animate(img_path, wav_path, out_mp4, 1.2, size, settings, "push_in", scene_id="hare_sprints")
                info = media.validate_video(ffprobe, out_mp4, size, dur)
                self.assertEqual((info["width"], info["height"]), size)

    def test_i2v_backend_budget_and_one_click_notebook(self):
        ffmpeg = media.find_tool("ffmpeg")
        nb_path = ROOT / "notebooks/one_click_i2v_test.ipynb"
        raw = nb_path.read_text(encoding="utf-8")
        self.assertLess(nb_path.stat().st_size, 32 * 1024)
        self.assertNotIn("EMBEDDED_PACKAGE_B64", raw)
        nb = json.loads(raw)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])

        profile = validate_i2v_profile(json.loads((ROOT / "configs/one_click_i2v_test.json").read_text(encoding="utf-8")))
        check_i2v_hardware(profile, cuda=True, bf16=True, vram_gib=24.0, ram_gib=32.0, disk_gib=40.0)

        pkg = ROOT / "examples/i2v_hare_sprints_v1_one_click.zip"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            info = extract_and_verify_i2v_bundle(pkg, work / "bundles")

            def fake_loader(prof, cache_dir):
                def pipe(**kwargs):
                    self.assertEqual(kwargs["width"], 832)
                    self.assertEqual(kwargs["height"], 480)
                    self.assertEqual(kwargs["num_frames"], 81)
                    return SimpleNamespace(frames=[["frame"]])

                def export_video(frames, out_path, fps=16):
                    _make_dummy_mp4(Path(out_path), ffmpeg, size=(832, 480), duration=0.5)

                return pipe, export_video

            created = []

            def backend_factory(settings, prev):
                b = WanI2VBackend(settings, info["profile"], work / "cache", prev, pipeline_loader=fake_loader)
                created.append(b)
                return b

            out = run_one_click_i2v_batch(
                info["bundle_dir"], work, backend_factory=backend_factory, cache_dir=work / "cache"
            )
            self.assertEqual(out["outcome"]["status"], "complete")
            self.assertEqual(len(out["outcome"]["assets"]), 1)
            self.assertEqual(out["outcome"]["assets"][0]["target_id"], "hare_sprints")
            self.assertTrue(out["archive_path"].is_file())

            # Strict budget=1 guard: second clip attempt must fail immediately
            with self.assertRaises(RuntimeError):
                created[0].generate_clip(
                    info["bundle_dir"] / "keyframes/hare_sprints.png", "p", "n", work / "extra.mp4", seed=1
                )

    def test_episode_8_clip_animation_notebook_resume_and_partial_export(self):
        ffmpeg = media.find_tool("ffmpeg")
        nb_path = ROOT / "notebooks/one_click_episode_animation.ipynb"
        raw = nb_path.read_text(encoding="utf-8")
        self.assertLess(nb_path.stat().st_size, 32 * 1024)
        self.assertNotIn("EMBEDDED_PACKAGE_B64", raw)
        nb = json.loads(raw)
        self.assertEqual(nb["metadata"]["colab"]["gpuType"], "A100")
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])

        expected_shots = [
            "hook_blur", "crossroads_greeting", "go_signal", "stone_slip",
            "hare_sprints", "hare_helps", "crossing_together", "crossing_finish",
        ]
        pkg = ROOT / "examples/episode_01_i2v_v1_one_click.zip"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            info = extract_and_verify_i2v_bundle(pkg, work / "bundles")
            self.assertEqual([s["target_id"] for s in info["request"]["scenes"]], expected_shots)
            self.assertEqual(info["profile"]["max_new_clips"], 8)

            def fail_on_5th_loader(prof, cache_dir):
                calls = {"n": 0}
                def pipe(**kwargs):
                    calls["n"] += 1
                    if calls["n"] >= 5:
                        raise RuntimeError("Simulated OOM on 5th clip")
                    return SimpleNamespace(frames=[["frame"]])
                def export_video(frames, out_path, fps=16):
                    _make_dummy_mp4(Path(out_path), ffmpeg, size=(832, 480), duration=0.4)
                return pipe, export_video

            partial = run_one_click_i2v_batch(
                info["bundle_dir"], work,
                backend_factory=lambda s, p: WanI2VBackend(s, info["profile"], work / "cache", p, pipeline_loader=fail_on_5th_loader),
                cache_dir=work / "cache", allow_partial=True,
            )
            self.assertEqual(partial["outcome"]["status"], "partial")
            self.assertEqual(len(partial["outcome"]["assets"]), 4)
            self.assertEqual(len(partial["outcome"]["errors"]), 4)
            self.assertTrue(partial["archive_path"].is_file())

            # Resume: re-running skips the 4 already-generated clips and finishes the remaining 4
            generated_on_resume = {"n": 0}
            def resume_loader(prof, cache_dir):
                def pipe(**kwargs):
                    generated_on_resume["n"] += 1
                    return SimpleNamespace(frames=[["frame"]])
                def export_video(frames, out_path, fps=16):
                    _make_dummy_mp4(Path(out_path), ffmpeg, size=(832, 480), duration=0.4)
                return pipe, export_video

            resumed = run_one_click_i2v_batch(
                info["bundle_dir"], work,
                backend_factory=lambda s, p: WanI2VBackend(s, info["profile"], work / "cache", p, pipeline_loader=resume_loader),
                cache_dir=work / "cache", allow_partial=True,
            )
            self.assertEqual(resumed["outcome"]["status"], "complete")
            self.assertEqual(len(resumed["outcome"]["assets"]), 8)
            self.assertEqual(generated_on_resume["n"], 4)

    def test_i2v_init_failure_aborts_immediately_and_uses_wan_i2v_pipeline(self):
        src_text = (ROOT / "src/modern_video.py").read_text(encoding="utf-8")
        self.assertIn("WanImageToVideoPipeline.from_pretrained", src_text)
        self.assertIn("expand_timesteps=True", src_text)
        self.assertNotIn("DiffusionPipeline", src_text)

        nb_text = (ROOT / "notebooks/one_click_episode_animation.ipynb").read_text(encoding="utf-8")
        self.assertNotIn("ONLY 'hare_sprints'", nb_text)
        self.assertIn("animate 8 shot(s)", nb_text)

        pkg = ROOT / "examples/episode_01_i2v_v1_one_click.zip"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            info = extract_and_verify_i2v_bundle(pkg, work / "bundles")
            init_attempts = {"n": 0}

            def failing_factory(settings, prev):
                init_attempts["n"] += 1
                raise RuntimeError("Simulated pipeline init incompatibility")

            with self.assertRaises(RuntimeError):
                run_one_click_i2v_batch(
                    info["bundle_dir"], work, backend_factory=failing_factory, cache_dir=work / "cache", allow_partial=True
                )
            self.assertEqual(init_attempts["n"], 1)


if __name__ == "__main__":
    unittest.main()
