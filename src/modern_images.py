"""FLUX.2 Klein character provider. Model imports occur only inside Colab.

The distilled pipeline has no negative-prompt CFG control. Exclusions are
explicit prose instructions, recorded honestly alongside the effective prompt.
"""
import importlib.metadata
import platform
import re
import time


from .image_conditioning import CONDITIONING_REVISION, reference_layout


def validate_profile(profile):
    if (profile.get("version") != 1 or profile.get("model_id") != "black-forest-labs/FLUX.2-klein-4B" or
            not re.fullmatch(r"[a-f0-9]{40}", profile.get("model_revision", "")) or
            profile.get("model_license") != "apache-2.0" or profile.get("dtype") != "bfloat16" or
            profile.get("negative_prompt_mode") != "inline_instruction_constraints_no_cfg"):
        raise ValueError("Unsupported or unpinned one-click model/license/conditioning profile")
    settings = profile.get("settings", {})
    mode = profile.get("mode", "references")
    if (settings.get("family") != "flux2_klein" or settings.get("steps") != 4 or
            settings.get("reference_size") != 1024 or profile.get("guidance_scale") != 1.0):
        raise ValueError("Prepared one-click profile requires flux2_klein, four steps, 1024px and guidance 1.0")
    if mode == "references":
        if profile.get("max_new_images") != 2 or set(profile.get("characters", {})) != {"hare", "tortoise"}:
            raise ValueError("Prepared one-click profile requires four steps, 1024px and a strict two-reference budget")
    elif mode == "challenge_scene":
        if profile.get("max_new_images") != 1 or set(profile.get("scenes", {})) != {"challenge"}:
            raise ValueError("Prepared one-click challenge profile requires a strict one-scene budget for 'challenge'")
    else:
        raise ValueError(f"Unsupported one-click profile mode: {mode}")
    for key, floor in (("min_vram_gib", 22), ("min_ram_gib", 24), ("min_disk_gib", 24)):
        if isinstance(profile.get(key), bool) or not isinstance(profile.get(key), (int, float)) or profile[key] < floor:
            raise ValueError(f"Unsafe hardware requirement: {key}")
    return profile


def check_hardware(profile, *, cuda, bf16, vram_gib, ram_gib, disk_gib):
    validate_profile(profile)
    instruction = "Select Runtime > Change runtime type > L4 or A100 GPU with High-RAM, then Run all. T4/CPU are not supported by this prepared BF16 profile."
    if not cuda or not bf16 or vram_gib < profile["min_vram_gib"]:
        raise RuntimeError("No compatible CUDA/BF16 GPU (need at least 22 GiB VRAM). " + instruction)
    if ram_gib < profile["min_ram_gib"]:
        raise RuntimeError("Insufficient host RAM (need at least 24 GiB for unquantized CPU offload). " + instruction)
    if disk_gib < profile["min_disk_gib"]:
        raise RuntimeError("Need at least 24 GiB free disk for checkpoint/cache. Free runtime disk; no weights downloaded.")


def effective_reference_prompt(job):
    if job["kind"] != "character":
        raise ValueError("This milestone generates character references only; scene generation is disabled")
    return job["prompt"] + "\nDo not include any of the following: " + job["negative_prompt"] + "."


def effective_scene_prompt(job, layout):
    if job["kind"] != "scene":
        raise ValueError("Expected a scene generation job")
    order = layout["reference_order"]
    if order == ["hare", "tortoise"]:
        binding = (
            "Use the provided character reference images to keep exact character appearance and 3D render style: "
            "reference image 1 is the white rabbit in the royal-blue sleeveless vest (place on the left side of the path); "
            "reference image 2 is the green tortoise with the smooth rounded chocolate-brown shell (place on the right side of the path). "
        )
    else:
        binding = f"Use the provided character reference images in order ({', '.join(order)}) to keep exact character appearance and 3D render style. "
    return binding + job["prompt"] + "\nDo not include any of the following: " + job["negative_prompt"] + "."


class FluxKleinBackend:
    def __init__(self, settings, profile, cache_dir, previous=None):
        validate_profile(profile)
        if any(getattr(settings, key) != value for key, value in profile["settings"].items()):
            raise ValueError("Worker settings differ from the pinned model profile")
        if previous and any(previous.get(key) != profile[key] for key in ("model_id", "model_revision", "model_license")):
            raise ValueError("Persistent model provenance changed; use a NEW batch")
        try:
            import google.colab
        except ImportError as exc:
            raise RuntimeError("This model backend is Colab-only; no local model loading") from exc
        import torch
        from diffusers import Flux2KleinPipeline
        from huggingface_hub import HfApi
        from pathlib import Path
        import shutil
        ram_gib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                           if line.startswith("MemTotal:"))) / 1024**2
        cuda = torch.cuda.is_available()
        check_hardware(profile, cuda=cuda, bf16=cuda and torch.cuda.get_device_capability(0)[0] >= 8,
            vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0,
            ram_gib=ram_gib, disk_gib=shutil.disk_usage(cache_dir).free / 2**30)
        # Public metadata only before weights; never log in or accept click-through terms.
        info = HfApi(token=False).model_info(profile["model_id"], revision=profile["model_revision"])
        if info.gated or info.private or getattr(info.card_data, "license", None) != profile["model_license"]:
            raise RuntimeError("Checkpoint access/license changed. Review it explicitly; no automatic authorization or substitute model.")
        self.torch, self.settings, self.profile = torch, settings, profile
        self.provenance = {key: profile[key] for key in ("model_id", "model_revision", "model_license", "license_url", "negative_prompt_mode")}
        self.provenance.update(gpu=torch.cuda.get_device_name(0), vram_GiB=torch.cuda.get_device_properties(0).total_memory / 2**30,
            dtype="bfloat16", cpu_offload=True, python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            versions={name: importlib.metadata.version(name) for name in ("diffusers", "transformers", "accelerate", "huggingface-hub", "safetensors", "Pillow")})
        start = time.perf_counter()
        try:
            self.pipe = Flux2KleinPipeline.from_pretrained(profile["model_id"], revision=profile["model_revision"],
                torch_dtype=torch.bfloat16, use_safetensors=True, low_cpu_mem_usage=True,
                cache_dir=str(cache_dir), token=False)
            self.pipe.enable_model_cpu_offload()
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CUDA/host OOM while loading Klein. No retry or fallback. Select A100/High-RAM; preserve any partial export.") from exc
            raise RuntimeError("Klein model loading failed; no retry or substitute model. Check network, disk and pinned dependencies. " + str(exc)) from exc
        self.provenance.update(model_load_seconds=round(time.perf_counter()-start, 3),
            scheduler={"class": type(self.pipe.scheduler).__name__, "config": dict(self.pipe.scheduler.config)})

    def generate(self, job, references):
        mode = self.profile.get("mode", "references")
        if job["kind"] == "character":
            if references:
                raise ValueError("This reference-only provider does not silently apply or omit scene reference conditioning")
            prompt = effective_reference_prompt(job)
            conditioning = "text_only"
            layout = None
            ref_images = None
        elif job["kind"] == "scene" and mode == "challenge_scene":
            if not references:
                raise ValueError("Scene generation requires approved character reference images")
            layout = reference_layout(references)
            prompt = effective_scene_prompt(job, layout)
            conditioning = CONDITIONING_REVISION
            ref_images = [references[c] for c in layout["reference_order"]]
        else:
            raise ValueError("This provider profile does not allow unconfigured scene generation")
        # Match the pipeline's Qwen chat template; reject truncation before inference.
        text = self.pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        count = len(self.pipe.tokenizer(text, truncation=False)["input_ids"])
        if count > 512:
            raise ValueError(f"Effective prompt exceeds 512 tokens ({count}); no inference attempted")
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        call_kwargs = dict(
            prompt=prompt,
            height=job["size"][1],
            width=job["size"][0],
            guidance_scale=1.0,
            num_inference_steps=4,
            num_images_per_prompt=1,
            max_sequence_length=512,
            generator=torch.Generator(device="cpu").manual_seed(job["seed"]),
        )
        if ref_images is not None:
            call_kwargs["image"] = ref_images
        try:
            with torch.inference_mode():
                result = self.pipe(**call_kwargs)
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CUDA OOM during generation. No automatic retry. Export partial results, then use A100/High-RAM.") from exc
            raise
        meta = {
            **self.provenance,
            "conditioning": conditioning,
            "effective_prompt": prompt,
            "prompt_token_count": count,
            "inference_steps": 4,
            "guidance_scale": 1.0,
            "peak_cuda_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }
        if layout is not None:
            meta["conditioning_layout"] = layout
            meta["reference_conditioning_mode"] = "flux2_klein_multi_reference_images"
        return result.images[0], meta
