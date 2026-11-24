"""Tests for the one-click FLUX.2-klein-4B character-reference notebook and worker.

All tests run locally on CPU using mocked/synthetic inference without downloading
model weights or requiring CUDA/Colab packages.
"""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from PIL import Image

from scripts.build_one_click_notebook import build as build_one_click_notebook
from src.asset_store import AssetStore
from src.assets import import_batch
from src.batch_handoff import validate_result
from src.characters import load_bible
from src.colab_images import ColabDiffusersBackend
from src.image_batch import BatchSettings, BatchWorker, scene_seed
from src.modern_images import (
    FluxKleinBackend,
    check_hardware,
    effective_reference_prompt,
    validate_profile,
)
from src.one_click import (
    DEFAULT_PACKAGE_NAME,
    ROOT,
    extract_and_verify_bundle,
    load_profile,
    prepare_one_click_handoff,
    run_one_click_batch,
    validate_one_click_result,
)
from src.production_state import file_hash
from src.schema import load_story


class SyntheticKleinBackend:
    def __init__(self, profile, fail_id=None):
        self.profile = profile
        self.fail_id = fail_id
        self.calls = []
        self.provenance = {
            "model_id": profile["model_id"],
            "model_revision": profile["model_revision"],
            "model_license": profile["model_license"],
            "license_url": profile["license_url"],
            "negative_prompt_mode": profile["negative_prompt_mode"],
            "dtype": "bfloat16",
            "cpu_offload": True,
        }

    def generate(self, job, references):
        if references:
            raise ValueError("References must be empty for character reference generation")
        self.calls.append((job["kind"], job["id"], job["seed"]))
        if job["id"] == self.fail_id:
            raise RuntimeError("Synthetic Klein inference failure")
        prompt = effective_reference_prompt(job)
        return Image.new("RGB", job["size"], "#8ab4f8"), {
            **self.provenance,
            "conditioning": "text_only",
            "effective_prompt": prompt,
            "prompt_token_count": 120,
            "inference_steps": 4,
            "guidance_scale": 1.0,
            "peak_cuda_allocated_GiB": 12.5,
        }


class OneClickGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile, self.profile_bytes, self.profile_sha256 = load_profile()

    def test_profile_and_v4_character_specifications(self):
        validate_profile(self.profile)
        self.assertEqual(self.profile["model_id"], "black-forest-labs/FLUX.2-klein-4B")
        self.assertEqual(self.profile["model_license"], "apache-2.0")
        self.assertEqual(self.profile["max_new_images"], 2)
        self.assertEqual(self.profile["settings"]["family"], "flux2_klein")
        self.assertEqual(self.profile["settings"]["steps"], 4)
        self.assertEqual(self.profile["settings"]["reference_size"], 1024)

        hare_spec = self.profile["characters"]["hare"]
        for required in ("Exactly one white rabbit", "pink inner ears", "small pink nose",
                         "royal-blue sleeveless vest", "without neckwear", "without lower-body clothing",
                         "turned 45 degrees"):
            self.assertIn(required, hare_spec["subject"])
        for excluded in ("bow tie", "shorts", "trousers", "accessories", "frontal pose", "duplicate animals"):
            self.assertIn(excluded, hare_spec["negative_prompt"])

        tortoise_spec = self.profile["characters"]["tortoise"]
        for required in ("Exactly one friendly green tortoise", "four legs", "leaf-green skin",
                         "smooth rounded dome-shaped shell", "BROWN", "hexagonal scutes", "turned 45 degrees"):
            self.assertIn(required, tortoise_spec["subject"])
        for excluded in ("gray skin", "olive shell", "green shell", "stacked shell", "repeated tortoises"):
            self.assertIn(excluded, tortoise_spec["negative_prompt"])

    def test_hardware_preflight_blocks_missing_gpu_low_vram_low_ram_and_low_disk(self):
        check_hardware(self.profile, cuda=True, bf16=True, vram_gib=22.0, ram_gib=25.0, disk_gib=30.0)
        with self.assertRaisesRegex(RuntimeError, "No compatible CUDA/BF16 GPU"):
            check_hardware(self.profile, cuda=False, bf16=False, vram_gib=0, ram_gib=30, disk_gib=30)
        with self.assertRaisesRegex(RuntimeError, "No compatible CUDA/BF16 GPU"):
            check_hardware(self.profile, cuda=True, bf16=False, vram_gib=15.0, ram_gib=30, disk_gib=30)
        with self.assertRaisesRegex(RuntimeError, "Insufficient host RAM"):
            check_hardware(self.profile, cuda=True, bf16=True, vram_gib=22.5, ram_gib=12.0, disk_gib=30)
        with self.assertRaisesRegex(RuntimeError, "free disk"):
            check_hardware(self.profile, cuda=True, bf16=True, vram_gib=22.5, ram_gib=30.0, disk_gib=10.0)

    def test_prepare_extract_and_run_one_click_batch_end_to_end_mocked(self):
        handoff = prepare_one_click_handoff(self.root / "handoff")
        self.assertEqual(handoff["reference_ids"], ["hare", "tortoise"])
        self.assertEqual(handoff["scene_ids"], [])

        bundle_info = extract_and_verify_bundle(
            handoff["package_path"],
            self.root / "bundles",
            expected_package_sha256=handoff["package_sha256"],
        )
        self.assertEqual(bundle_info["request_sha256"], handoff["request_sha256"])
        self.assertEqual(bundle_info["profile_sha256"], handoff["profile_sha256"])

        backend = SyntheticKleinBackend(self.profile)
        loads = 0

        def factory(settings, previous):
            nonlocal loads
            loads += 1
            return backend

        work_root = self.root / "colab_temp"
        result = run_one_click_batch(bundle_info["bundle_dir"], work_root, backend_factory=factory)
        self.assertEqual(loads, 1)
        self.assertEqual([c[1] for c in backend.calls], ["hare", "tortoise"])
        self.assertTrue(result["contact_sheet_path"].is_file())
        self.assertTrue(result["archive_path"].is_file())
        self.assertEqual(result["validation"]["status"], "complete")
        self.assertEqual(result["validation"]["completed_reference_ids"], ["hare", "tortoise"])
        self.assertEqual(result["validation"]["completed_scene_ids"], [])

        # Both standalone one-click validation and local batch_handoff.validate_result pass:
        self.assertEqual(
            validate_result(result["manifest_path"])["completed_reference_ids"],
            ["hare", "tortoise"],
        )
        self.assertEqual(
            validate_one_click_result(result["manifest_path"])["completed_reference_ids"],
            ["hare", "tortoise"],
        )

        # Verify result ZIP contents:
        with zipfile.ZipFile(result["archive_path"]) as zf:
            names = set(zf.namelist())
            for expected in ("manifest.json", "request.json", "generation_profile.json"):
                self.assertIn(expected, names)
            self.assertEqual(sum(n.startswith("images/character_") and n.endswith(".png") for n in names), 2)
            self.assertEqual(sum(n.startswith("reviews/contact_sheet_") and n.endswith(".jpg") for n in names), 1)

        # Verify local import into AssetStore keeps candidates UNAPPROVED:
        store = AssetStore(self.root / "library")
        story = load_story(ROOT / "scripts/tortoise_and_hare.json")
        bible = load_bible(ROOT / "configs/character_bible.json")
        imported = import_batch(store, result["manifest_path"], story, bible)
        self.assertEqual(len(imported), 2)
        self.assertEqual(store.read()["approvals"], {"scene": {}, "character": {}})

        # Verify idempotent rerun in same work_root does not reload model or regenerate images:
        again = run_one_click_batch(bundle_info["bundle_dir"], work_root, backend_factory=factory)
        self.assertEqual(loads, 1)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(again["validation"]["completed_reference_ids"], ["hare", "tortoise"])

    def test_flux_klein_backend_mocked_pipeline_and_legacy_sdxl_preservation(self):
        with self.assertRaisesRegex(ValueError, "legacy backend supports only SDXL/SD15"):
            ColabDiffusersBackend(BatchSettings(family="flux2_klein"))

        backend = FluxKleinBackend.__new__(FluxKleinBackend)
        backend.settings = BatchSettings(family="flux2_klein", steps=4, base_seed=20261126, reference_size=1024)
        backend.profile = self.profile
        backend.provenance = {
            "model_id": self.profile["model_id"],
            "model_revision": self.profile["model_revision"],
            "model_license": self.profile["model_license"],
            "license_url": self.profile["license_url"],
            "negative_prompt_mode": self.profile["negative_prompt_mode"],
        }
        backend.torch = MagicMock()
        backend.torch.cuda.max_memory_allocated.return_value = 8 * 2**30
        backend.pipe = MagicMock()
        backend.pipe.tokenizer.apply_chat_template.return_value = "templated prompt"
        backend.pipe.tokenizer.return_value = {"input_ids": list(range(95))}
        backend.pipe.return_value = SimpleNamespace(images=[Image.new("RGB", (1024, 1024), "white")])

        job = {
            "kind": "character",
            "id": "hare",
            "prompt": "One white rabbit in a blue vest.",
            "negative_prompt": "bow tie, shorts",
            "seed": scene_seed("hare", 20261126),
            "size": (1024, 1024),
        }
        image, meta = backend.generate(job, {})
        self.assertEqual(image.size, (1024, 1024))
        self.assertEqual(meta["conditioning"], "text_only")
        self.assertIn("Do not include any of the following: bow tie, shorts.", meta["effective_prompt"])
        call_kwargs = backend.pipe.call_args.kwargs
        self.assertEqual(call_kwargs["num_inference_steps"], 4)
        self.assertEqual(call_kwargs["guidance_scale"], 1.0)
        self.assertEqual(call_kwargs["width"], 1024)
        self.assertEqual(call_kwargs["height"], 1024)

        with self.assertRaisesRegex(ValueError, "reference-only provider"):
            backend.generate(job, {"hare": Image.new("RGB", (64, 64))})

        backend.pipe.tokenizer.return_value = {"input_ids": list(range(600))}
        with self.assertRaisesRegex(ValueError, "exceeds 512 tokens"):
            backend.generate(job, {})

    def test_one_click_notebook_structure_ordering_and_simulated_execution(self):
        notebook_path = ROOT / "notebooks/one_click_generation.ipynb"
        self.assertTrue(notebook_path.is_file())
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 5)

        assignments = {}
        for idx, cell in enumerate(code_cells, start=1):
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            compile(source, f"one_click_cell_{idx}", "exec")
            self.assertTrue(source.lstrip().startswith(f"# {idx}."))
            for statement in ast.parse(source).body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    try:
                        assignments[statement.targets[0].id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        pass

        self.assertFalse(assignments["USE_GOOGLE_DRIVE"])
        self.assertEqual(assignments["PERSISTENT_DIR"], "")
        self.assertEqual(assignments["RESULT_BATCH_ID"], "")
        self.assertEqual(assignments["EXPECTED_PACKAGE_SHA256"], file_hash(ROOT / f"examples/{DEFAULT_PACKAGE_NAME}.zip"))

        # Simulate executing Cells 1, 2, 3, 4, 5 in order in a mocked Colab environment:
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda idx: (8, 9),
            get_device_properties=lambda idx: SimpleNamespace(total_memory=int(22.5 * 2**30)),
            get_device_name=lambda idx: "NVIDIA L4",
        )
        downloads = []
        displayed = []
        fake_colab = ModuleType("google.colab")
        fake_colab.files = SimpleNamespace(download=lambda path: downloads.append(path))
        fake_colab.drive = SimpleNamespace(mount=lambda path: (_ for _ in ()).throw(AssertionError("Drive must not mount by default")))
        fake_google = ModuleType("google")
        fake_google.colab = fake_colab
        fake_ipython_display = ModuleType("IPython.display")
        fake_ipython_display.display = lambda obj: displayed.append(obj)

        work_dir = self.root / "notebook_sim"
        env = {}
        c1 = "".join(code_cells[0]["source"]).replace('PERSISTENT_DIR = ""', f'PERSISTENT_DIR = {json.dumps(str(work_dir))}')
        c2 = "".join(code_cells[1]["source"]).replace(
            'meminfo = Path("/proc/meminfo")',
            'meminfo = WORK_ROOT / "fake_meminfo"\nmeminfo.write_text("MemTotal: 52428800 kB\\n")',
        )
        c3 = "".join(code_cells[2]["source"])
        c4 = "".join(code_cells[3]["source"])
        c5 = "".join(code_cells[4]["source"])

        req_pins = {
            "diffusers": "0.37.1",
            "transformers": "4.57.6",
            "accelerate": "1.12.0",
            "huggingface-hub": "0.36.2",
            "safetensors": "0.7.0",
            "Pillow": "12.1.1",
        }
        synthetic_backend = SyntheticKleinBackend(self.profile)

        prev_sys_path = list(sys.path)
        prev_src_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
        try:
            with patch.dict(sys.modules, {
                "torch": fake_torch,
                "google": fake_google,
                "google.colab": fake_colab,
                "IPython.display": fake_ipython_display,
            }), patch("urllib.request.urlopen", side_effect=OSError("offline test uses embedded fallback")), \
                    patch("importlib.metadata.version", side_effect=lambda pkg: req_pins[pkg]):
                exec(compile(c1, "cell1", "exec"), env)
                exec(compile(c2, "cell2", "exec"), env)
                exec(compile(c3, "cell3", "exec"), env)
                exec(compile(c4, "cell4", "exec"), env)
                with patch("src.modern_images.FluxKleinBackend", return_value=synthetic_backend):
                    exec(compile(c5, "cell5", "exec"), env)
        finally:
            sys.path[:] = prev_sys_path
            for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
                sys.modules.pop(k, None)
            sys.modules.update(prev_src_modules)

        self.assertEqual(len(downloads), 1)
        self.assertTrue(Path(downloads[0]).is_file())
        self.assertEqual(len(displayed), 3)  # 1 contact sheet + 2 character images
        self.assertEqual([c[1] for c in synthetic_backend.calls], ["hare", "tortoise"])


if __name__ == "__main__":
    unittest.main()
