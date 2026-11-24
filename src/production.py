"""Resumable master production: python -m src.production --script story.json --output-dir NEW_FOLDER."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import uuid

from PIL import Image

from .asset_store import AssetStore
from .batch_handoff import create_handoff, receive_result, prepare_existing_production, MODULES
from .assets import export_request
from .characters import load_bible
from .image_providers import FileImageProvider
from .main import load_config
from .media import find_tool, mix_music, validate_video
from .narration import wav_duration
from .production_plan import SIZES, StructuredStoryPlanner, load_settings
from .production_providers import FFmpegAnimation, FFmpegFinalVideo, HybridAnimation, OfflineVoice, source_revision
from .production_state import ProductionState, file_hash, production_directory, write_json
from .schema import load_story

ROOT = Path(__file__).resolve().parents[1]


def asset_inventory(store, story):
    """Report all relevant candidate states without selecting an unapproved file."""
    catalog = store.read()
    relevant = {"scene": {s.reuse_asset or s.id for s in story.scenes}, "character": {c for s in story.scenes for c in s.characters}}
    records = []
    for kind, targets in relevant.items():
        for target in sorted(targets):
            candidates = sorted(r["asset_id"] for r in catalog["assets"].values()
                                if r["kind"] == kind and r["target_id"] == target)
            pin = catalog["approvals"][kind].get(target)
            row = {"kind": kind, "id": target, "candidate_ids": candidates,
                   "awaiting_approval": [c for c in candidates if c != pin], "approval_pin": pin,
                   "status": "awaiting_approval" if candidates else "missing"}
            try:
                approved = store.approved(kind, target)
                if approved:
                    record, path = approved
                    row.update(status="approved", asset_id=record["asset_id"], sha256=record["sha256"], path=str(path))
            except (OSError, ValueError, KeyError) as exc:
                row.update(status="invalid_approval", error=str(exc), manual_unlock_required=True)
            records.append(row)
    return records


class ColabRequestPreparer:
    """Request-only adapter; filters damaged approvals without bypassing locks."""
    cache_key = source_revision("assets.py", "batch_handoff.py", *MODULES) + "_production_request_v2"

    def __init__(self, store):
        self.store = store

    def prepare(self, story, bible, missing_ids, destination):
        store = self.store
        invalid = []

        class CheckedApprovals:
            def approved(self, kind, target):
                try:
                    return store.approved(kind, target)
                except (OSError, ValueError, KeyError) as exc:
                    invalid.append({"kind": kind, "id": target, "error": str(exc)})
                    return None

        # Reuse the existing notebook request format and approved-reference exporter.
        export_request(CheckedApprovals(), story, bible, destination, missing_ids or [story.scenes[0].id])
        request = json.loads(destination.read_text(encoding="utf-8"))
        if not missing_ids:
            request["scenes"] = []
        catalog = store.read()
        # A damaged pin must still be manually unlocked before a replacement job.
        request["locked_scene_ids"] = sorted(set(missing_ids) & set(catalog["approvals"]["scene"]))
        blocked_refs = [c["id"] for c in request["characters"] if "approved_reference_png_base64" not in c]
        request["production_notes"] = {"generation_started": False, "missing_scene_ids": missing_ids,
            "references_needing_approval_or_generation": blocked_refs, "invalid_approvals": invalid,
            "manual_unlock_required": request["locked_scene_ids"]}
        write_json(destination, request)
        handoff = create_handoff(destination, store, destination.parent / "image_batch_handoff.zip")
        return {"path": str(destination), "sha256": file_hash(destination), "scene_ids": missing_ids,
                "references_needing_approval_or_generation": blocked_refs, "generation_started": False,
                "handoff": handoff}


def produce(script: Path, output_dir: Path, *, settings_path: Path | None = None,
            pipeline_path: Path = ROOT / "configs/pipeline.json",
            bible_path: Path = ROOT / "configs/character_bible.json",
            asset_dir: Path = ROOT / "assets/library", ffmpeg=None, ffprobe=None,
            planner=None, images=None, voice=None, animation=None, renderer=None,
            request_preparer=None, refresh_narration=False):
    """Coordinate providers. Injection is supported for future implementations/tests.

    Provider cache_key values must change whenever their generated behavior changes.
    Existing src.main is intentionally independent and remains functional.
    """
    with production_directory(output_dir) as root:
        state = ProductionState(root)
        current_scene = None
        try:
            state.data["input"] = {"script_path": str(Path(script).resolve()),
                "settings_path": str(Path(settings_path).resolve()) if settings_path else None,
                "bible_path": str(Path(bible_path).resolve()), "asset_directory": str(Path(asset_dir).resolve())}
            state.save()
            story = load_story(Path(script))
            bible = load_bible(Path(bible_path))
            settings = load_settings(Path(settings_path) if settings_path else None)
            pipeline = load_config(Path(pipeline_path))
            store = AssetStore(asset_dir)
            images = images or FileImageProvider(store)
            planner = planner or StructuredStoryPlanner()
            plan, effective_bible = planner.plan(story, bible, settings, images, pipeline)
            inventory = asset_inventory(store, story)
            state.data.update(title=story.title, settings=settings,
                input={**state.data["input"], "script_sha256": file_hash(script), "script": json.loads(Path(script).read_text(encoding="utf-8")),
                       "bible_sha256": file_hash(bible_path), "character_bible": asdict(effective_bible),
                       "pipeline_path": str(Path(pipeline_path).resolve()), "pipeline": pipeline},
                plan={k: v for k, v in plan.items() if k != "scenes"}, scenes=plan["scenes"],
                warnings=plan["warnings"], assets=inventory)
            missing = [s["id"] for s in plan["scenes"] if s["formats"] and s["image_selection"]["provider"] != "approved_file"]
            state.data["asset_summary"] = {"approved_image_scene_ids": [s["id"] for s in plan["scenes"] if s["formats"] and s["id"] not in missing],
                "placeholder_scene_ids": missing, "needs_ai_generation": missing,
                "awaiting_approval": sorted(c for a in inventory for c in a["awaiting_approval"])}
            state.save()
            request_preparer = request_preparer or ColabRequestPreparer(store)

            def prepare_request(directory):
                path = directory / "ai_generation_request.json"
                result = request_preparer.prepare(story, effective_bible, missing, path)
                artifacts = [path]
                if result.get("handoff"):
                    artifacts.append(Path(result["handoff"]["package_path"]))
                return result, artifacts

            request, action = state.stage("ai_request", {"provider": request_preparer.cache_key,
                "story": [asdict(s) for s in story.scenes], "bible": asdict(effective_bible),
                "assets": inventory, "missing_scene_ids": missing}, prepare_request)
            state.data["ai_generation_request"] = {**request, "action": action}
            if request.get("handoff"):
                import shutil
                package = request["handoff"]
                handoff = root / "image_handoff" / f"request_{request['sha256'][:12]}_{package['package_sha256'][:12]}.zip"
                handoff.parent.mkdir(exist_ok=True)
                if handoff.exists():
                    if file_hash(handoff) != package["package_sha256"]:
                        raise ValueError("Published image handoff was changed; refusing to overwrite it")
                else:
                    with open(package["package_path"], "rb") as source, handoff.open("xb") as target:
                        shutil.copyfileobj(source, target)
                state.data["image_batch_handoff"] = {**package, "package_path": str(handoff),
                    "manual_colab_start_required": True}
            state.data["active_stage"] = "media_preflight"
            state.save()
            ffmpeg = find_tool("ffmpeg", ffmpeg or pipeline.get("ffmpeg"))
            ffprobe = find_tool("ffprobe", ffprobe or pipeline.get("ffprobe"))
            voice = voice or OfflineVoice()
            has_approved_clips = bool(store.read()["approvals"].get("clip"))
            if animation is None:
                animation = HybridAnimation(ffmpeg, store) if (settings["animation"] == "hybrid" or has_approved_clips) else FFmpegAnimation(ffmpeg)
            renderer = renderer or FFmpegFinalVideo(ffmpeg, ffprobe)
            image_revision = getattr(images, "cache_key", source_revision("image_providers.py", "visuals.py"))
            state.data["providers"] = {"planner": planner.cache_key, "images": image_revision, "voice": voice.cache_key,
                "animation": animation.cache_key, "final_video": renderer.cache_key, "requests": request_preparer.cache_key}
            state.save()
            clips = {name: [] for name in settings["formats"]}
            durations = {name: 0.0 for name in clips}
            speech_seconds = 0.0
            if refresh_narration or "voice_cache_revision" not in state.data:
                state.data["voice_cache_revision"] = uuid.uuid4().hex if refresh_narration else "default"
                state.save()
            refresh_token = state.data["voice_cache_revision"]
            for scene, planned in zip(story.scenes, state.data["scenes"]):
                current_scene = planned
                if not planned["formats"]:
                    continue
                planned["status"] = "running"
                state.save()
                print(f"[{planned['index']}/{len(story.scenes)}] {scene.id}", flush=True)

                def narrate(directory):
                    speech = voice.synthesize(scene, directory, settings["narration"])
                    result = {"path": str(speech.path.resolve()), "sha256": file_hash(speech.path),
                        "duration": speech.duration, "backend": speech.backend, "warnings": list(speech.warnings)}
                    return result, [speech.path, *directory.glob("*.txt")]

                def validate_speech(result):
                    if abs(wav_duration(Path(result["path"])) - result["duration"]) > .01:
                        raise ValueError("Narration duration differs from WAV")

                narration, voice_action = state.stage(f"voice:{scene.id}", {"provider": voice.cache_key,
                    "text": scene.narration, "scene_id": scene.id, "minimum_duration": scene.duration,
                    "language": settings["language"], "settings": settings["narration"], "refresh": refresh_token}, narrate, validate_speech)
                planned["narration"] = {**narration, "status": "completed", "action": voice_action}
                state.data["warnings"].extend(f"{scene.id}: {w}" for w in narration["warnings"])
                speech_seconds += narration["duration"]
                # Measured audio drives timing; pause_after adds a deliberate beat after narration.
                duration = max(scene.duration, narration["duration"] + (pipeline["tail_seconds"] if narration["backend"] != "silence" else 0) + scene.pause_after)
                duration = math.ceil(duration * pipeline["fps"]) / pipeline["fps"]
                planned["render_duration"] = duration
                state.save()
                for name in planned["formats"]:
                    size = SIZES[name]

                    def draw(directory):
                        # Guard against approval changes between planning and rendering.
                        if images.resolve(scene) != planned["image_selection"]:
                            raise RuntimeError("Asset selection changed during production; rerun to make a fresh plan")
                        path = directory / f"{scene.id}_{name}.png"
                        result = images.render(scene, story.title, planned["index"], len(story.scenes), path, size)
                        if result["provider"] != planned["image_selection"]["provider"]:
                            raise RuntimeError("Image provider changed its selection; rerun to plan the fallback explicitly")
                        with Image.open(path) as image:
                            image.load()
                            if image.size != size:
                                raise ValueError("Image provider returned the wrong dimensions")
                        return {**result, "path": str(path), "sha256": file_hash(path)}, [path]

                    # Recheck approval protection even when a prepared image is cached.
                    if images.resolve(scene) != planned["image_selection"]:
                        raise RuntimeError("Asset selection changed during production; rerun")
                    visual, visual_action = state.stage(f"image:{scene.id}:{name}", {"provider": image_revision,
                        "scene": asdict(scene), "title": story.title, "index": planned["index"], "count": len(story.scenes),
                        "selection": planned["image_selection"], "size": size}, draw)
                    planned["images"][name] = {**visual, "status": "completed", "action": visual_action}

                    def animate(directory):
                        path = directory / f"{scene.id}_{name}.mp4"
                        if hasattr(animation, "clip_cache_key"):
                            actual = animation.animate(
                                Path(visual["path"]), Path(narration["path"]), path, duration, size, pipeline, scene.motion, scene_id=scene.id
                            )
                        else:
                            actual = animation.animate(Path(visual["path"]), Path(narration["path"]), path, duration, size, pipeline, scene.motion)
                        validation = validate_video(ffprobe, path, size, actual)
                        return {"path": str(path), "sha256": file_hash(path), "duration": actual, "validation": validation}, [path]

                    clip_deps = {"provider": animation.cache_key,
                        "image_sha256": visual["sha256"], "audio_sha256": narration["sha256"], "duration": duration,
                        "size": size, "fps": pipeline["fps"], "threads": pipeline["threads"], "animation": settings["animation"],
                        "motion": scene.motion}
                    if hasattr(animation, "clip_cache_key"):
                        clip_deps["approved_clip"] = animation.clip_cache_key(scene.id)
                    clip, clip_action = state.stage(f"clip:{scene.id}:{name}", clip_deps, animate)
                    clip_mode = "i2v_clip" if (hasattr(animation, "clip_cache_key") and animation.clip_cache_key(scene.id) != "camera_only") else "camera_only"
                    planned["clips"][name] = {**clip, "status": "completed", "action": clip_action, "animation_mode": clip_mode}
                    clips[name].append(clip)
                    durations[name] += clip["duration"]
                    state.save()
                planned["status"] = "completed"
                state.save()
            current_scene = None
            i2v_shots = sorted({
                s["id"] for s in state.data["scenes"]
                if any(c.get("animation_mode") == "i2v_clip" for c in s.get("clips", {}).values())
            })
            cam_shots = sorted({
                s["id"] for s in state.data["scenes"]
                if s.get("clips") and s["id"] not in i2v_shots
            })
            state.data["animation_coverage"] = {"i2v_clip_scene_ids": i2v_shots, "camera_only_scene_ids": cam_shots}
            for name, records in clips.items():
                def assemble(directory):
                    path = directory / f"{name}.mp4"
                    validation = renderer.assemble([Path(c["path"]) for c in records], path, SIZES[name], durations[name])
                    return {**validation, "path": str(path), "sha256": file_hash(path)}, [path]

                result, action = state.stage(f"final:{name}", {"provider": renderer.cache_key,
                    "ordered_clips": [{"sha256": r["sha256"], "duration": r["duration"]} for r in records],
                    "size": SIZES[name]}, assemble)
                state.data["outputs"][name] = {**result, "status": "completed", "action": action}
                state.save()
                music = settings["music"]
                if music:
                    def mix(directory, name=name, result=result):
                        path = directory / f"{name}_music.mp4"
                        mix_music(ffmpeg, Path(result["path"]), Path(music["path"]), path, music["volume_db"], duration=result["duration"])
                        check = renderer.validate(path, SIZES[name], result["duration"])
                        return {**check, "path": str(path), "sha256": file_hash(path), "music": {k: music[k] for k in ("license", "source_url", "volume_db")},
                                "narration_only_path": result["path"]}, [path]

                    mixed, mix_action = state.stage(f"music:{name}", {"final": result["sha256"], "music_sha256": file_hash(music["path"]),
                        "volume_db": music["volume_db"], "license": music["license"]}, mix)
                    state.data["outputs"][name] = {**mixed, "status": "completed", "action": mix_action}
                    state.save()
            rendered_seconds = state.data["outputs"].get("landscape", next(iter(state.data["outputs"].values())))["duration"]
            target = plan["target_duration"]
            for label, seconds in (("Narration", speech_seconds), ("Rendered video", rendered_seconds)):
                if abs(seconds - target) > max(5, target * .2):
                    state.data["warnings"].append(f"{label} is {seconds:.2f}s versus target {target:.2f}s. No automatic stretching or scene invention was applied.")
            state.data["warnings"] = list(dict.fromkeys(state.data["warnings"]))
            state.data["timing"] = {"target_seconds": target, "requested_scene_seconds": plan["requested_scene_seconds"],
                "narration_seconds": speech_seconds, "rendered_primary_seconds": rendered_seconds}
            spoken = all(s["narration"].get("backend") != "silence" for s in state.data["scenes"] if s["formats"])
            blockers = []
            if missing:
                blockers.append(f"Placeholder cards still used for scenes: {missing}")
            if not spoken:
                blockers.append("Narration is silent for at least one scene")
            window = settings["duration_range"]
            if window and not window[0] <= rendered_seconds <= window[1]:
                blockers.append(f"Rendered duration {rendered_seconds:.1f}s is outside the required {window[0]}-{window[1]}s range")
                state.data["warnings"].append(blockers[-1])
            state.data["quality"] = {"uses_placeholders": bool(missing), "spoken_narration": spoken,
                "ai_generation_started": False, "publish_ready": not blockers, "publish_blockers": blockers,
                "human_review_required": True,
                "animation": "camera_motion_only; no character animation is rendered by this pipeline"}
            state.finish()
            return state.data
        except BaseException as exc:
            if current_scene is not None:
                current_scene.update(status="failed", error=str(exc))
                stage = state.data["active_stage"].split(":")
                failure = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}
                if stage[0] == "voice":
                    current_scene["narration"].update(failure)
                elif len(stage) == 3 and stage[0] in {"image", "clip"}:
                    current_scene["images" if stage[0] == "image" else "clips"].setdefault(stage[2], {}).update(failure)
            state.finish(exc)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/production.json")
    parser.add_argument("--pipeline-config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--character-bible", type=Path, default=ROOT / "configs/character_bible.json")
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--import-image-batch", type=Path, help="Validate/import a result ZIP or manifest and create contact sheets; does not render or approve")
    parser.add_argument("--prepare-image-batch", action="store_true", help="Export the existing production request with worker and references, without rendering")
    parser.add_argument("--handoff-dir", type=Path, help="Destination folder for read-only handoff export")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--refresh-narration", action="store_true", help="Retry speech, including cached silent fallbacks")
    args = parser.parse_args()
    try:
        if args.prepare_image_batch and args.import_image_batch:
            parser.error("Choose either prepare or import")
        if args.prepare_image_batch:
            result = prepare_existing_production(args.output_dir, args.handoff_dir or args.output_dir / "image_handoff")
            print(json.dumps(result, indent=2))
            return
        if args.import_image_batch:
            result = receive_result(args.output_dir, args.import_image_batch, asset_dir=args.asset_dir)
            print(json.dumps(result, indent=2))
            return
        if not args.script:
            parser.error("--script is required unless preparing/importing an image batch")
        result = produce(args.script, args.output_dir, settings_path=args.settings, pipeline_path=args.pipeline_config,
            bible_path=args.character_bible, asset_dir=args.asset_dir or ROOT / "assets/library", ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
            refresh_narration=args.refresh_narration)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\nSee the production manifest for progress/failure details.\n")
    last = result["runs"][-1]
    print(json.dumps({"status": result["status"], "outputs": result["outputs"],
        "asset_summary": result["asset_summary"], "ai_generation_request": result["ai_generation_request"],
        "image_batch_handoff": result.get("image_batch_handoff"),
        "executed_stages": len(last["executed_stages"]), "reused_stages": len(last["reused_stages"]),
        "warnings": result["warnings"], "manifest": str((args.output_dir / "production_manifest.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
