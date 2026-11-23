"""Deterministic Pillow cartoon cards; no models, network, or external assets."""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from .schema import Scene


def font(size: int):
    candidates = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf",
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def wrap(draw, text, face, width):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=face) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def draw_tortoise(d, x, y, scale=1.0):
    def box(coords):
        return tuple((x if i % 2 == 0 else y) + v * scale for i, v in enumerate(coords))
    ink = "#244b42"
    for a in (-68, 35):
        d.ellipse(box((a, 20, a + 40, 62)), fill="#8fbd63", outline=ink, width=3)
    d.ellipse(box((-100, -55, 65, 48)), fill="#43885a", outline=ink, width=4)
    d.arc(box((-72, -44, 37, 55)), 180, 355, fill="#a8d477", width=5)
    d.line([box((-30, -50, -30, 43))[:2], box((-30, -50, -30, 43))[2:]], fill="#a8d477", width=4)
    d.ellipse(box((38, -18, 115, 42)), fill="#abd17a", outline=ink, width=3)
    d.ellipse(box((83, -2, 93, 9)), fill=ink)
    d.arc(box((78, 8, 102, 29)), 5, 140, fill=ink, width=3)


def draw_hare(d, x, y, scale=1.0, sleeping=False):
    def b(coords):
        return tuple((x if i % 2 == 0 else y) + v * scale for i, v in enumerate(coords))
    ink, fur = "#684a68", "#f1dce9"
    d.ellipse(b((-60, -30, 40, 90)), fill=fur, outline=ink, width=3)
    d.ellipse(b((-78, 60, 55, 95)), fill=fur, outline=ink, width=3)
    d.ellipse(b((-83, 1, -40, 44)), fill="white", outline=ink, width=3)
    for a in (-20, 17):
        d.ellipse(b((a, -172, a + 28, -63)), fill=fur, outline=ink, width=3)
        d.ellipse(b((a + 8, -156, a + 21, -83)), fill="#e5abc5")
    d.ellipse(b((-38, -86, 55, 3)), fill=fur, outline=ink, width=3)
    if sleeping:
        d.arc(b((14, -52, 35, -32)), 0, 180, fill=ink, width=3)
    else:
        d.ellipse(b((22, -53, 32, -39)), fill=ink)
    d.ellipse(b((45, -30, 62, -19)), fill="#bc7292")
    d.arc(b((23, -31, 48, -9)), 0, 125, fill=ink, width=3)


def render_card(scene: Scene, title: str, index: int, count: int, path: Path, size=(1280, 720)):
    """Composition adapts to portrait so captions and characters stay visible."""
    w, h = size
    portrait = h > w
    image = Image.new("RGB", size, "#def1f4")
    d = ImageDraw.Draw(image)
    sky_bottom = int(h * (0.65 if portrait else 0.64))
    d.ellipse((w * .77, h * .14, w * .77 + 86, h * .14 + 86), fill="#ffce65")
    for cx, cy in ((w * .14, h * .22), (w * .55, h * .17)):
        for dx, dy in ((0, 0), (32, -17), (66, 0)):
            d.ellipse((cx + dx, cy + dy, cx + dx + 68, cy + dy + 40), fill="#ffffff")
    d.ellipse((-w * .2, h * .36, w * .8, h * .99), fill="#bddb9b")
    d.ellipse((w * .35, h * .31, w * 1.3, h * .99), fill="#9ec983")
    d.rectangle((0, sky_bottom, w, h), fill="#92bb75")
    d.polygon([(w * .43, h * .42), (w * .50, h * .42), (w * .90, h * .78), (w * .10, h * .78)], fill="#efd49c")
    tree_x, tree_y = w * .84, h * .49
    d.rounded_rectangle((tree_x - 17, tree_y - 104, tree_x + 17, tree_y + 62), radius=10, fill="#98715a")
    for dx, dy in ((-70, -145), (-12, -175), (30, -127)):
        d.ellipse((tree_x + dx, tree_y + dy, tree_x + dx + 92, tree_y + dy + 100), fill="#569265")
    for n in range(12):
        fx, fy = int(w * (.07 + .077 * n)), int(h * (.67 + .035 * (n % 2)))
        d.line((fx, fy, fx, fy + 13), fill="#4b8053", width=3)
        d.ellipse((fx - 5, fy - 5, fx + 5, fy + 5), fill="#fff1a8" if n % 2 else "#f4b3ac")
    y = h * (.58 if portrait else .59)
    scale = 1.05 if portrait else 1.15
    if "tortoise" in scene.characters:
        tx = w * (.59 if scene.id == "finish" else .32)
        draw_tortoise(d, tx, y, scale)
    if "hare" in scene.characters:
        hx = w * (.28 if scene.id == "finish" else .75 if scene.id == "racing_ahead" else .65)
        draw_hare(d, hx, y - 6, scale, sleeping=scene.id == "nap")
        if scene.id == "nap":
            d.text((hx + 54, y - 155), "z z z", font=font(32), fill="#684a68")
    if scene.id == "finish":
        d.line((w * .72, y - 100, w * .72, y + 100), fill="#fdfbf2", width=8)
        d.text((w * .56, y - 205), "FINISH!", font=font(34), fill="#244b42")
    margin = 38
    d.rounded_rectangle((margin, 30, w - margin, 106 if portrait else 94), radius=22, fill="#fff9e9")
    title_size = 31 if portrait else 35
    while d.textlength(title, font=font(title_size)) > w - 2 * margin - 38 and title_size > 12:
        title_size -= 1
    d.text((margin + 20, 47), title, font=font(title_size), fill="#244b42")
    d.text((margin + 5, 121 if portrait else 108), f"STORY TIME   /   {index:02d} OF {count:02d}", font=font(20), fill="#366457")
    card_top = int(h * (.735 if portrait else .765))
    d.rounded_rectangle((margin, card_top, w - margin, h - 28), radius=24, fill="#fffaf0")
    text_width, available_h = w - 2 * margin - 42, h - 28 - card_top - 40
    size_pt = 33 if portrait else 30
    while True:
        face = font(size_pt)
        lines = wrap(d, scene.narration, face, text_width)
        line_h = size_pt + 9
        if len(lines) * line_h <= available_h and all(d.textlength(line, font=face) <= text_width for line in lines):
            break
        size_pt -= 1
        if size_pt < 10:
            raise ValueError(f"Scene {scene.id}: narration too long for a readable card; split into shorter scenes")
    for n, line in enumerate(lines):
        d.text((margin + 21, card_top + 19 + n * line_h), line, font=face, fill="#263f3c")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
