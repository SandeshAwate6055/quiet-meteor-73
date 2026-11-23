"""Local asset import/review/approval and Colab request export. No network access."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .asset_store import AssetStore, digest, read_image
from .characters import load_bible, safe_relative, scene_prompt
from .schema import load_story
from .visuals import font

ROOT = Path(__file__).resolve().parents[1]


def import_batch(store, manifest_path, story, bible):
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("assets"), list):
        raise ValueError("Import requires a version 1 manifest with assets list")
    scene_ids = {s.id for s in story.scenes}
    prepared = []
    root = manifest_path.parent.resolve()
    for entry in data["assets"]:
        kind, target = entry["kind"], entry["target_id"]
        known = scene_ids if kind == "scene" else set(bible.characters) if kind == "character" else set()
        if target not in known:
            raise ValueError(f"Unknown asset target: {kind}/{target}")
        source = (root / safe_relative(entry["file"])).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Import file escapes batch folder")
        if digest(source.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Import checksum mismatch: {source}")
        if not isinstance(entry.get("metadata", {}), dict):
            raise ValueError("Asset metadata must be an object")
        read_image(source)
        prepared.append((entry, source))
    # Validate the complete batch before importing. Import never changes approvals.
    # Retain the batch-level provenance that would otherwise be lost on import.
    batch_provenance = {"batch_id": data.get("batch_id") or root.name, "manifest_path": str(manifest_path.resolve()),
                        "manifest_sha256": digest(manifest_path.read_bytes()),
                        "metadata": {key: value for key, value in data.items() if key != "assets"}}
    return [store.import_image(source, entry["kind"], entry["target_id"],
                              {**entry.get("metadata", {}), "source_batch": {
                                  **batch_provenance, "source_file": entry["file"], "source_sha256": entry["sha256"]}})
            for entry, source in prepared]


def review(store, destination):
    catalog = store.read()
    rows = sorted(catalog["assets"].values(), key=lambda a: (a["kind"], a["target_id"], a["asset_id"]))
    destination.mkdir(parents=True, exist_ok=True)
    # Pages keep review memory bounded for growing libraries.
    pages = []
    for start in range(0, len(rows), 12):
        group = rows[start:start + 12]
        sheet = Image.new("RGB", (1000, 300 * ((len(group) + 2) // 3)), "#fffaf0")
        draw = ImageDraw.Draw(sheet)
        for n, row in enumerate(group):
            x, y = (n % 3) * 333, (n // 3) * 300
            try:
                thumb = ImageOps.contain(read_image(store.path_for(row)), (315, 210))
                sheet.paste(thumb, (x + (333 - thumb.width) // 2, y + 4))
            except (OSError, ValueError) as exc:
                draw.text((x + 8, y + 8), f"Unavailable: {str(exc)[:30]}", fill="red", font=font(14))
            locked = catalog["approvals"][row["kind"]].get(row["target_id"]) == row["asset_id"]
            label = f"#{start+n+1} {row['kind']}/{row['target_id']}\n{row['sha256'][:16]} {'LOCKED' if locked else 'candidate'}"
            draw.multiline_text((x + 8, y + 220), label, font=font(17), fill="#263f3c")
        page = destination / f"contact_sheet_{start // 12 + 1:03d}.jpg"
        sheet.save(page)
        pages.append(str(page.resolve()))
    listing = [{"review_number": n + 1, **row,
                "approved": catalog["approvals"][row["kind"]].get(row["target_id"]) == row["asset_id"]}
               for n, row in enumerate(rows)]
    (destination / "review.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
    return {"pages": pages, "assets": listing}


def export_request(store, story, bible, destination, selected=None):
    selected = selected or [s.id for s in story.scenes if s.id in ("challenge", "racing_ahead", "steady_steps", "finish")]
    if not selected or len(set(selected)) != len(selected) or set(selected) - {s.id for s in story.scenes}:
        raise ValueError("Select unique scene ids present in the story")
    request = {"version": 1, "title": story.title, "style": bible.style, "environment": bible.environment,
               "characters": [], "scenes": [], "locked_scene_ids": [], "negative_prompt":
               "text, watermark, logo, scary, violence, photorealistic, deformed, extra limbs, duplicate characters"}
    for character in bible.characters.values():
        row = {"id": character.id, "name": character.name, "appearance": character.appearance,
               "reference_image_paths": list(character.reference_image_paths), "metadata": character.metadata,
               "prompt": f"{bible.style}. Character reference, exactly one {character.name}: {character.appearance}. Full body, three-quarter view, neutral pale background, no lettering."}
        approved = store.approved("character", character.id)
        if approved:
            record, path = approved
            # Reusable approved references travel inside the JSON request, not as unsafe absolute paths.
            image = ImageOps.contain(read_image(path), (768, 768))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            row["approved_reference_png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
            row["approved_asset_id"] = record["asset_id"]
            row["approved_reference_sha256"] = digest(buffer.getvalue())
        request["characters"].append(row)
    for scene in story.scenes:
        if scene.id in selected:
            request["scenes"].append({"id": scene.id, "characters": list(scene.characters), "prompt": scene_prompt(scene, bible)})
            if store.approved("scene", scene.id):
                request["locked_scene_ids"].append(scene.id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(request, indent=2), encoding="utf-8")
    return str(destination.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets/library")
    parser.add_argument("--script", type=Path, default=ROOT / "scripts/tortoise_and_hare.json")
    parser.add_argument("--bible", type=Path, default=ROOT / "configs/character_bible.json")
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import-batch")
    imp.add_argument("manifest", type=Path)
    rev = commands.add_parser("review")
    rev.add_argument("--output-dir", type=Path, default=ROOT / "outputs/asset_review")
    approve = commands.add_parser("approve")
    approve.add_argument("asset_id")
    unlock = commands.add_parser("unlock")
    unlock.add_argument("kind", choices=("scene", "character"))
    unlock.add_argument("target_id")
    unlock.add_argument("--reason", required=True)
    export = commands.add_parser("export-request")
    export.add_argument("--output", type=Path, default=ROOT / "notebooks/colab_input.json")
    export.add_argument("--scene-ids", nargs="+")
    args = parser.parse_args()
    store = AssetStore(args.asset_dir)
    try:
        if args.command == "import-batch":
            result = import_batch(store, args.manifest, load_story(args.script), load_bible(args.bible))
        elif args.command == "review":
            result = review(store, args.output_dir)
        elif args.command == "approve":
            store.approve(args.asset_id)
            result = {"approved": args.asset_id}
        elif args.command == "unlock":
            store.unlock(args.kind, args.target_id, args.reason)
            result = {"unlocked": args.target_id}
        else:
            result = export_request(store, load_story(args.script), load_bible(args.bible), args.output, args.scene_ids)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
