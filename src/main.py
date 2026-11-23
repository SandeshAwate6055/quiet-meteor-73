"""Run from the project root: python -m src.main --script scripts/tortoise_and_hare.json."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

from .media import concatenate, find_tool, render_scene, run, validate_video
from .narration import synthesize
from .schema import load_story
from .asset_store import AssetStore
from .characters import load_bible, scene_prompt
from .image_providers import FileImageProvider, PillowProvider

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    for key, low, high in (("fps", 12, 60), ("threads", 1, 4), ("speech_rate", -10, 10)):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"Config {key} must be an integer between {low} and {high}")
    tail = config.get("tail_seconds")
    if isinstance(tail, bool) or not isinstance(tail, (int, float)) or not math.isfinite(tail) or not 0 <= tail <= 5:
        raise ValueError("Config tail_seconds must be between 0 and 5")
    return config


def build(args):
    story = load_story(args.script)
    config = load_config(args.config)
    bible = load_bible(args.character_bible)
    for scene in story.scenes:
        scene_prompt(scene, bible)
    provider = PillowProvider() if args.image_provider == "pillow" else FileImageProvider(AssetStore(args.asset_dir))
    ffmpeg = find_tool("ffmpeg", args.ffmpeg or config.get("ffmpeg"))
    ffprobe = find_tool("ffprobe", args.ffprobe or config.get("ffprobe"))
    # Fail before any lengthy work if binaries cannot load their local DLLs.
    run([ffmpeg, "-version"], 15)
    run([ffprobe, "-version"], 15)
    selected = args.vertical_scenes or config.get("vertical_scene_ids") or [s.id for s in story.scenes[-3:]]
    if not isinstance(selected, list) or not selected or any(not isinstance(s, str) for s in selected) or len(set(selected)) != len(selected):
        raise ValueError("Vertical selection must be a nonempty list of unique scene ids")
    unknown = set(selected) - {s.id for s in story.scenes}
    if unknown:
        # Default selection is sample-specific; a new story gets its last three scenes.
        if args.vertical_scenes:
            raise ValueError(f"Unknown vertical scenes: {sorted(unknown)}")
        selected = [s.id for s in story.scenes[-3:]]
    slug = re.sub(r"[^a-z0-9]+", "_", story.title.lower()).strip("_") or "story"
    out = args.output_dir.resolve() / slug
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"title": story.title, "script": str(args.script.resolve()), "scenes": [], "warnings": [], "outputs": {}}
    landscape_clips, portrait_clips = [], []
    total, vertical_total = 0.0, 0.0
    for index, scene in enumerate(story.scenes, 1):
        print(f"[{index}/{len(story.scenes)}] Narrating and drawing {scene.id}", flush=True)
        speech = synthesize(scene, out / "audio", config["speech_rate"], args.silent)
        for warning in speech.warnings:
            print(f"WARNING [{scene.id}]: {warning}", file=sys.stderr, flush=True)
            manifest["warnings"].append(f"{scene.id}: {warning}")
        duration = max(scene.duration, speech.duration + (config["tail_seconds"] if speech.backend != "silence" else 0))
        image = out / "images" / f"{scene.id}_landscape.png"
        visual = provider.render(scene, story.title, index, len(story.scenes), image, (1280, 720))
        clip = out / f"{index:02d}_{scene.id}_landscape.mp4"
        actual = render_scene(ffmpeg, image, speech.path, clip, duration, (1280, 720), config["fps"], config["threads"])
        landscape_clips.append(clip)
        total += actual
        record = {"id": scene.id, "requested_duration": scene.duration, "speech_duration": speech.duration,
                  "render_duration": actual, "narration_backend": speech.backend,
                  "visual_description": scene.visual_description, "characters": list(scene.characters),
                  "images": {"landscape": visual}}
        manifest["scenes"].append(record)
        if scene.id in selected:
            image = out / "images" / f"{scene.id}_vertical.png"
            record["images"]["vertical"] = provider.render(scene, story.title, index, len(story.scenes), image, (720, 1280))
            clip = out / f"{index:02d}_{scene.id}_vertical.mp4"
            vertical_total += render_scene(ffmpeg, image, speech.path, clip, duration, (720, 1280), config["fps"], config["threads"])
            portrait_clips.append(clip)
    for name, clips, size, expected in (("landscape", landscape_clips, (1280, 720), total),
                                        ("vertical", portrait_clips, (720, 1280), vertical_total)):
        path = out / f"{name}.mp4"
        print(f"Assembling and validating {path.name}", flush=True)
        concatenate(ffmpeg, clips, path)
        manifest["outputs"][name] = validate_video(ffprobe, path, size, expected)
        # Full decode catches corruption that metadata-only inspection cannot detect.
        run([ffmpeg, "-v", "error", "-xerror", "-threads", "2", "-i", path, "-f", "null", "-"], 120)
    manifest["spoken_narration"] = all(s["narration_backend"] != "silence" for s in manifest["scenes"])
    manifest["sample_target_met"] = 30 <= manifest["outputs"]["landscape"]["duration"] <= 45
    if not manifest["sample_target_met"]:
        manifest["warnings"].append("Landscape is outside the 30–45 second target; shorten script/durations or increase speech_rate.")
    report = out / "manifest.json"
    report.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["outputs"], indent=2), flush=True)
    print(f"Report: {report}\nSpoken narration: {manifest['spoken_narration']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build local cartoon story videos with Pillow, SAPI and FFmpeg")
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase2")
    parser.add_argument("--character-bible", type=Path, default=ROOT / "configs/character_bible.json")
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets/library")
    parser.add_argument("--image-provider", choices=("auto", "pillow"), default="auto")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--vertical-scenes", nargs="+", help="Scene ids to include, in story order")
    parser.add_argument("--silent", action="store_true", help="Exercise the explicit silent fallback")
    args = parser.parse_args()
    try:
        build(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
