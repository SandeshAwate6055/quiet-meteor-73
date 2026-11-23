"""Colab-only Diffusers adapter, adapted from the original Phase 2 notebook.

Importing this module does not import/install Torch or download weights.
"""
import platform
import time


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
        from PIL import Image, ImageOps
        torch = self.torch
        options = {}
        conditioning = "text_only" if job["kind"] == "character" else "not_applicable"
        if references:
            self._adapter()
            self.pipe.set_ip_adapter_scale(self.settings.adapter_scale)
            board = Image.new("RGB", (512, 512), "#f4f4ee")
            cell = 512 // len(references)
            for index, image in enumerate(references.values()):
                fitted = ImageOps.contain(image, (max(8, cell - 12), 500))
                board.paste(fitted, (index * cell + (cell - fitted.width) // 2, (512 - fitted.height) // 2))
            options["ip_adapter_image"] = board
            conditioning = "ip_adapter_contact_sheet"
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
            "adapter_id": self.provenance["adapter_id"] if references else None,
            "adapter_revision": self.provenance["adapter_revision"] if references else None,
            "adapter_weight": self.provenance.get("adapter_weight") if references else None,
            "adapter_scale": self.settings.adapter_scale if references else None,
            "inference_steps": self.settings.steps, "guidance_scale": 6.5 if self.settings.family == "sdxl" else 7.5,
            "peak_cuda_allocated_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
