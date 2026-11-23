"""Colab-only Diffusers adapter, adapted from the original Phase 2 notebook.

Importing this module does not import/install Torch or download weights.
"""
import platform
import time

from .image_conditioning import CONDITIONING_REVISION, reference_layout, reference_masks


class ColabDiffusersBackend:
    def __init__(self, settings, previous=None):
        try:
            import google.colab
        except ImportError as exc:
            raise RuntimeError("GPU backend is Colab-only. Local planning/import does not load models.") from exc
        import importlib.metadata
        import torch
        from diffusers import AutoPipelineForText2Image
        from huggingface_hub import HfApi
        if not torch.cuda.is_available():
            raise RuntimeError("No Colab CUDA GPU. Choose a GPU runtime explicitly.")
        self.torch, self.settings = torch, settings
        self.gpu_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        model_id = "stabilityai/stable-diffusion-xl-base-1.0" if settings.family == "sdxl" else "stable-diffusion-v1-5/stable-diffusion-v1-5"
        api = HfApi()
        previous = previous or {}
        model_revision = previous.get("model_revision") or api.model_info(model_id).sha
        adapter_revision = previous.get("adapter_revision") or api.model_info("h94/IP-Adapter").sha
        self.provenance = {"model_id": model_id, "model_revision": model_revision,
            "model_license": "CreativeML Open RAIL++-M" if settings.family == "sdxl" else "CreativeML OpenRAIL-M",
            "adapter_id": "h94/IP-Adapter", "adapter_revision": adapter_revision, "adapter_license": "Apache-2.0",
            "gpu": torch.cuda.get_device_name(0), "vram_GiB": self.gpu_gib,
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "versions": {n: importlib.metadata.version(n) for n in ("diffusers", "transformers", "accelerate", "safetensors", "huggingface-hub", "Pillow")}}
        kwargs = dict(torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True,
                      revision=model_revision, cache_dir="/content/story_model_cache")
        if settings.family == "sdxl":
            kwargs["variant"] = "fp16"
        start = time.perf_counter()
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self._offload()
        self.adapter_loaded = False
        self.provenance["model_load_seconds"] = round(time.perf_counter() - start, 3)
        self.provenance["scheduler"] = {"class": type(self.pipe.scheduler).__name__, "config": dict(self.pipe.scheduler.config)}

    def _offload(self):
        if self.gpu_gib < 8:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.enable_model_cpu_offload()

    def _adapter(self):
        if self.adapter_loaded:
            return
        self.pipe.remove_all_hooks()
        folder = "sdxl_models" if self.settings.family == "sdxl" else "models"
        weight = "ip-adapter_sdxl_vit-h.safetensors" if self.settings.family == "sdxl" else "ip-adapter_sd15.safetensors"
        # Matching architecture + ViT-H encoder. Any load error propagates to
        # the worker's persisted failure record; conditioning is never disabled.
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder=folder, weight_name=weight,
            image_encoder_folder="models/image_encoder", revision=self.provenance["adapter_revision"],
            cache_dir="/content/story_model_cache", low_cpu_mem_usage=True)
        self._offload()
        self.adapter_loaded = True
        self.provenance["adapter_weight"] = f"{folder}/{weight}"

    def generate(self, job, references):
        from PIL import Image
        torch = self.torch
        options = {}
        token_counts = {}
        for name in ("tokenizer", "tokenizer_2"):
            tokenizer = getattr(self.pipe, name, None)
            if tokenizer is None:
                continue
            counts = {label: len(tokenizer(text, truncation=False)["input_ids"])
                      for label, text in (("positive", job["prompt"]), ("negative", job["negative_prompt"]))}
            limit = tokenizer.model_max_length
            if any(count > limit for count in counts.values()):
                raise ValueError(f"{name} prompt would be truncated ({counts}, limit {limit}); shorten it before inference")
            token_counts[name] = {**counts, "limit": limit}
        conditioning = "text_only" if job["kind"] == "character" else "not_applicable"
        layout, mask_hashes = None, {}
        if references:
            from diffusers.image_processor import IPAdapterMaskProcessor
            self._adapter()
            layout = reference_layout(references)
            masks, mask_hashes = reference_masks(layout, job["size"])
            self.pipe.set_ip_adapter_scale([[self.settings.adapter_scale] * len(references)])
            # One adapter, N separately encoded images: outer list is adapters;
            # inner list is subjects. Masks are attention gates, not scene pixels.
            options["ip_adapter_image"] = [[references[c] for c in layout["reference_order"]]]
            w, h = job["size"]
            tensor = IPAdapterMaskProcessor().preprocess(masks, height=h, width=w)
            tensor = tensor.reshape(1, len(masks), h, w).to(device=self.pipe._execution_device, dtype=self.pipe.unet.dtype)
            options["cross_attention_kwargs"] = {"ip_adapter_masks": [tensor]}
            conditioning = CONDITIONING_REVISION
        elif self.adapter_loaded:
            if job["kind"] == "character":
                raise RuntimeError("New references must be generated before scenes in this worker invocation")
            # A scene without characters has no character conditioning requirement.
            self.pipe.set_ip_adapter_scale(0.0)
            options["ip_adapter_image"] = Image.new("RGB", (512, 512), "white")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = self.pipe(prompt=job["prompt"], negative_prompt=job["negative_prompt"],
                generator=torch.Generator(device="cpu").manual_seed(job["seed"]),
                width=job["size"][0], height=job["size"][1], num_inference_steps=self.settings.steps,
                guidance_scale=6.5 if self.settings.family == "sdxl" else 7.5, num_images_per_prompt=1, **options)
        if getattr(output, "nsfw_content_detected", None) and any(output.nsfw_content_detected):
            raise RuntimeError("Pipeline flagged this image; candidate was not exported")
        return output.images[0], {**self.provenance, "conditioning": conditioning,
            "conditioning_layout": layout, "mask_sha256": mask_hashes, "prompt_token_counts": token_counts,
            "adapter_id": self.provenance["adapter_id"] if references else None,
            "adapter_revision": self.provenance["adapter_revision"] if references else None,
            "adapter_weight": self.provenance.get("adapter_weight") if references else None,
            "adapter_scale": self.settings.adapter_scale if references else None,
            "inference_steps": self.settings.steps, "guidance_scale": 6.5 if self.settings.family == "sdxl" else 7.5,
            "peak_cuda_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
