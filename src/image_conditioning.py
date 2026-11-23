"""Small deterministic spatial-attention plan; no collage, Torch, or model loads."""
from PIL import Image, ImageDraw
import hashlib

CONDITIONING_REVISION = "ip_adapter_masked_references_v2"


def reference_layout(identities):
    ordered = sorted(identities)
    if len(set(ordered)) != len(ordered) or len(ordered) > 2:
        raise ValueError("This smoke configuration supports at most two distinct character references")
    boxes = ((.10, .22, .46, .94), (.54, .32, .92, .94)) if len(ordered) == 2 else ((.20, .20, .80, .94),)
    return {"revision": CONDITIONING_REVISION, "reference_order": ordered,
            "regions": {identity: list(box) for identity, box in zip(ordered, boxes)}}


def reference_masks(layout, size):
    w, h = size
    masks, hashes = [], {}
    for identity in layout["reference_order"]:
        box = layout["regions"][identity]
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(tuple(round(c * (w if n % 2 == 0 else h)) for n, c in enumerate(box)),
                                              radius=round(min(w, h) * .06), fill=255)
        masks.append(mask)
        hashes[identity] = hashlib.sha256(mask.tobytes()).hexdigest()
    return masks, hashes
