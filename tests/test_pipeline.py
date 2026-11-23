import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.main import ROOT, load_config
from src.media import concatenate, find_tool, render_scene, run, validate_video
from src.narration import synthesize, wav_duration, write_silence
from src.schema import Scene, load_story
from src.visuals import render_card


class StoryTests(unittest.TestCase):
    def test_original_sample_has_target_timing(self):
        story = load_story(ROOT / "scripts/tortoise_and_hare.json")
        self.assertEqual(len(story.scenes), 5)
        self.assertTrue(30 <= sum(s.duration for s in story.scenes) <= 45)

    def test_reject_invalid_script_fields_and_unsafe_ids(self):
        data = json.loads((ROOT / "scripts/tortoise_and_hare.json").read_text())
        for field, value in (("duration", 0), ("duration", True), ("duration", float("nan")),
                             ("id", "../escape"), ("narration", " "), ("characters", ["hare", "hare"])):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as temp:
                changed = copy.deepcopy(data)
                changed["scenes"][0][field] = value
                path = Path(temp) / "bad.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_story(path)
        with tempfile.TemporaryDirectory() as temp:
            data["scenes"][1]["id"] = data["scenes"][0]["id"]
            path = Path(temp) / "duplicate.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_story(path)

    def test_config_rejects_unbounded_threads(self):
        with tempfile.TemporaryDirectory() as temp:
            config = json.loads((ROOT / "configs/pipeline.json").read_text())
            config["threads"] = 99
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)


class NarrationTests(unittest.TestCase):
    def test_broken_speech_falls_back_to_valid_silent_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = Scene("test", "A small step.", 0.5, "A meadow.")
            with patch("src.narration.subprocess.run", side_effect=subprocess.TimeoutExpired("speech", 45)):
                result = synthesize(scene, Path(temp))
            self.assertEqual(result.backend, "silence")
            self.assertGreaterEqual(len(result.warnings), 2)
            self.assertAlmostEqual(wav_duration(result.path), 0.5, places=3)


class RenderTests(unittest.TestCase):
    def test_portrait_and_landscape_cards(self):
        story = load_story(ROOT / "scripts/tortoise_and_hare.json")
        with tempfile.TemporaryDirectory() as temp:
            for size in ((1280, 720), (720, 1280)):
                path = Path(temp) / f"card_{size[0]}.png"
                render_card(story.scenes[0], story.title, 1, 5, path, size)
                with Image.open(path) as image:
                    self.assertEqual(image.size, size)
                    self.assertEqual(image.mode, "RGB")

    def test_real_ffmpeg_concat_and_validation(self):
        try:
            ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
        except RuntimeError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory(prefix="story integration ") as temp:
            folder = Path(temp)
            image, audio = folder / "image.png", folder / "audio.wav"
            Image.new("RGB", (320, 180), "#9ec983").save(image)
            write_silence(audio, 0.5)
            clips = []
            total = 0
            for n in range(2):
                path = folder / f"clip_{n}.mp4"
                total += render_scene(ffmpeg, image, audio, path, 0.6, (320, 180))
                clips.append(path)
            output = folder / "joined.mp4"
            concatenate(ffmpeg, clips, output)
            info = validate_video(ffprobe, output, (320, 180), total)
            self.assertEqual(info["audio_codec"], "aac")
            self.assertGreater(info["duration"], 1)
            with self.assertRaises(ValueError):
                validate_video(ffprobe, output, (720, 1280))
            run([ffmpeg, "-v", "error", "-xerror", "-i", output, "-f", "null", "-"])


if __name__ == "__main__":
    unittest.main()
