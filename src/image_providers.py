"""Scene-image provider contract; optional AI files have no runtime ML dependency."""
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .asset_store import AssetStore, read_image
from .schema import Scene
from .visuals import font, render_card, wrap


class SceneImageProvider(Protocol):
    def render(self, scene: Scene, title: str, index: int, count: int, path: Path, size: tuple[int, int]) -> dict: ...


class PillowProvider:
    def resolve(self, scene):
        return {"provider": "pillow", "fallback_reason": "Pillow provider explicitly selected"}

    def render(self, scene, title, index, count, path, size):
        render_card(scene, title, index, count, path, size)
        return {"provider": "pillow", "resolution": list(size)}


def compose_illustration(source, scene, path, size):
    """Contain the complete illustration, including in portrait; never crop characters."""
    image = read_image(source)
    w, h = size
    canvas = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(24))
    art_height = int(h * .75)
    fitted = ImageOps.contain(image, (w - 40, art_height - 32), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((w - fitted.width) // 2, (art_height - fitted.height) // 2 + 12))
    draw = ImageDraw.Draw(canvas)
    top, margin = int(h * .765), 28
    draw.rounded_rectangle((margin, top, w - margin, h - 24), radius=22, fill="#fffaf0")
    for pts in range(32, 8, -1):
        face = font(pts)
        lines = wrap(draw, scene.narration, face, w - 2 * margin - 40)
        if len(lines) * (pts + 8) <= h - 24 - top - 32 and all(draw.textlength(s, font=face) <= w - 2 * margin - 40 for s in lines):
            break
    else:
        raise ValueError("Narration too long for illustration caption; split scene")
    for n, line in enumerate(lines):
        draw.text((margin + 20, top + 16 + n * (pts + 8)), line, font=face, fill="#263f3c")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


class FileImageProvider:
    def __init__(self, store: AssetStore, fallback: SceneImageProvider | None = None):
        self.store = store
        self.fallback = fallback or PillowProvider()

    def resolve(self, scene):
        """Read-only, checksum-checked selection shared by planning and rendering."""
        warning = None
        try:
            approved = self.store.approved("scene", scene.id)
            if approved:
                record, source = approved
                return {"provider": "approved_file", "asset_id": record["asset_id"],
                        "sha256": record["sha256"], "source_resolution": record["resolution"], "source_path": str(source)}
        except (OSError, ValueError, KeyError) as exc:
            warning = f"Approved image unavailable; using Pillow: {exc}"
        return {"provider": "pillow", "fallback_reason": warning or "No approved image for this scene"}

    def render(self, scene, title, index, count, path, size):
        selection = self.resolve(scene)
        if selection["provider"] == "approved_file":
            try:
                compose_illustration(Path(selection["source_path"]), scene, path, size)
                return {key: value for key, value in selection.items() if key != "source_path"} | {"resolution": list(size)}
            except (OSError, ValueError) as exc:
                selection = {"fallback_reason": f"Approved image unavailable; using Pillow: {exc}"}
        result = self.fallback.render(scene, title, index, count, path, size)
        result["fallback_reason"] = selection["fallback_reason"]
        return result
