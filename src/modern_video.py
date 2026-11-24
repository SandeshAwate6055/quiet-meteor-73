"""Wan2.2-TI2V-5B image-to-video backend for Colab. Model imports occur only at runtime."""
import inspect
from pathlib import Path
import re

from PIL import Image, ImageOps


PINNED_I2V_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def validate_i2v_profile(profile):
    if (
        profile.get("version") != 1
        or profile.get("mode") != "i2v_clips"
        or profile.get("model_id") != PINNED_I2V_MODEL_ID
        or not re.fullmatch(r"[a-f0-9]{40}", profile.get("model_revision", ""))
        or profile.get("model_license") != "apache-2.0"
        or profile.get("dtype") != "bfloat16"
    ):
        raise ValueError("Unsupported or unpinned one-click I2V model/license profile")
    settings = profile.get("settings", {})
    if (
        settings.get("family") != "wan22_ti2v_5b"
        or settings.get("width") != 832
        or settings.get("height") != 480
        or settings.get("num_frames") != 81
        or settings.get("fps") != 16
        or not isinstance(settings.get("steps"), int)
        or not 10 <= settings["steps"] <= 60
        or not isinstance(settings.get("guidance_scale"), (int, float))
        or settings["guidance_scale"] <= 0
    ):
        raise ValueError("Prepared I2V profile requires wan22_ti2v_5b, 832x480, 81 frames at 16 fps")
    scenes = profile.get("scenes")
    if (
        not isinstance(scenes, dict)
        or not 1 <= len(scenes) <= 10
        or profile.get("max_new_clips") != len(scenes)
        or any(not isinstance(v, dict) or not v.get("prompt") or not v.get("negative_prompt") for v in scenes.values())
    ):
        raise ValueError("Prepared I2V profile requires 1-10 scenes and max_new_clips equal to the scene count")
    for key, floor in (("min_vram_gib", 22), ("min_ram_gib", 24), ("min_disk_gib", 24)):
        if isinstance(profile.get(key), bool) or not isinstance(profile.get(key), (int, float)) or profile[key] < floor:
            raise ValueError(f"Unsafe I2V hardware requirement: {key}")
    return profile


def check_i2v_hardware(profile, *, cuda, bf16, vram_gib, ram_gib, disk_gib):
    validate_i2v_profile(profile)
    instruction = "Select Runtime > Change runtime type > L4 or A100 GPU with High-RAM, then Run all. T4/CPU are not supported."
    if not cuda or not bf16 or vram_gib < profile["min_vram_gib"]:
        raise RuntimeError(f"No compatible CUDA/BF16 GPU (need at least {profile['min_vram_gib']} GiB VRAM). " + instruction)
    if ram_gib < profile["min_ram_gib"]:
        raise RuntimeError(f"Insufficient host RAM (need at least {profile['min_ram_gib']} GiB). " + instruction)
    if disk_gib < profile["min_disk_gib"]:
        raise RuntimeError(f"Need at least {profile['min_disk_gib']} GiB free disk for checkpoint/cache. No weights downloaded.")


class WanI2VBackend:
    """Bounded Colab-only Wan2.2-TI2V-5B backend with strict clip budget enforcement."""

    def __init__(self, settings, profile, cache_dir, previous=None, *, pipeline_loader=None):
        validate_i2v_profile(profile)
        if settings != profile["settings"]:
            raise ValueError("I2V worker settings differ from the pinned model profile")
        if previous and any(previous.get(k) != profile[k] for k in ("model_id", "model_revision", "model_license")):
            raise ValueError("Persistent I2V model provenance changed; use a NEW batch")
        self.settings = dict(settings)
        self.profile = profile
        self.cache_dir = Path(cache_dir)
        self._generated_count = 0
        self._export_to_video = None

        if pipeline_loader is not None:
            self.pipe, self._export_to_video = pipeline_loader(profile, self.cache_dir)
            return

        try:
            import google.colab  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("This I2V model backend is Colab-only; no local model loading") from exc

        import shutil
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from diffusers.utils import export_to_video

        ram_gib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                           if line.startswith("MemTotal:"))) / 1024**2
        cuda = torch.cuda.is_available()
        check_i2v_hardware(
            profile,
            cuda=cuda,
            bf16=cuda and torch.cuda.get_device_capability(0)[0] >= 8,
            vram_gib=(torch.cuda.get_device_properties(0).total_memory / 2**30) if cuda else 0.0,
            ram_gib=ram_gib,
            disk_gib=shutil.disk_usage(self.cache_dir).free / 2**30,
        )
        vae = AutoencoderKLWan.from_pretrained(
            profile["model_id"],
            revision=profile["model_revision"],
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=str(self.cache_dir),
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            profile["model_id"],
            revision=profile["model_revision"],
            vae=vae,
            expand_timesteps=True,
            torch_dtype=torch.bfloat16,
            cache_dir=str(self.cache_dir),
        )
        if "image" not in inspect.signature(pipe.__call__).parameters:
            raise RuntimeError(
                f"Loaded pipeline {type(pipe).__name__} does not accept 'image' conditioning; "
                "cannot run true first-frame I2V."
            )
        if not getattr(pipe.config, "expand_timesteps", False):
            raise RuntimeError("WanImageToVideoPipeline must have expand_timesteps=True for Wan2.2-TI2V-5B")
        pipe.enable_model_cpu_offload()
        self.pipe = pipe
        self._export_to_video = export_to_video

    def generate_clip(self, image_path: Path, prompt: str, negative_prompt: str, destination: Path, *, seed: int) -> dict:
        if self._generated_count >= self.profile["max_new_clips"]:
            raise RuntimeError(f"I2V clip budget exceeded ({self.profile['max_new_clips']}); refusing additional generation")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        w, h = self.settings["width"], self.settings["height"]
        with Image.open(image_path) as img:
            keyframe = ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), (w, h), method=Image.Resampling.LANCZOS)

        kwargs = {
            "image": keyframe,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": h,
            "width": w,
            "num_frames": self.settings["num_frames"],
            "num_inference_steps": self.settings["steps"],
            "guidance_scale": self.settings["guidance_scale"],
        }
        try:
            import torch
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(seed))
        except ImportError:
            pass

        result = self.pipe(**kwargs)
        frames = result.frames[0]
        self._export_to_video(frames, str(destination), fps=self.settings["fps"])
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"I2V pipeline produced empty output: {destination}")
        self._generated_count += 1
        return {
            "seed": int(seed),
            "width": w,
            "height": h,
            "num_frames": self.settings["num_frames"],
            "fps": self.settings["fps"],
            "steps": self.settings["steps"],
            "guidance_scale": self.settings["guidance_scale"],
            "model_id": self.profile["model_id"],
            "model_revision": self.profile["model_revision"],
            "model_license": self.profile["model_license"],
        }
