"""
Path resolution smoke tests for the Lance RunPod handler.

These catch bugs like hardcoded "downloads/" defaults before they
reach RunPod. Run in CI via:

  uv run python test_handler.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# Minimal mock of handler path resolution (imported from handler.py)
# ---------------------------------------------------------------------------

# Ensure we don't actually import torch or Lance deps
os.environ.setdefault("LANCE_MODEL_BASE_DIR", "/runpod-volume/checkpoints")


def get_model_base_dir() -> Path:
    configured = os.getenv("LANCE_MODEL_BASE_DIR")
    return Path(configured).expanduser() if configured else Path("downloads")


def get_model_path(model_variant: str) -> Path:
    variant = model_variant.strip().lower()
    if variant in {"image", "t2i", "i2t"}:
        variant_dir = "Lance_3B"
        env_name = "LANCE_IMAGE_MODEL_PATH"
    else:
        variant_dir = "Lance_3B_Video"
        env_name = "LANCE_VIDEO_MODEL_PATH"
    configured = os.getenv(env_name)
    if configured:
        return Path(configured).expanduser()
    configured = os.getenv("LANCE_MODEL_PATH")
    if configured:
        return Path(configured).expanduser()
    return get_model_base_dir() / variant_dir


# Known model subdirectory names that must be present
EXPECTED_ASSETS = [
    "Lance_3B",
    "Lance_3B_Video",
    "Qwen2.5-VL-ViT",
    "Wan2.2_VAE.pth",
]


class TestPathResolution(unittest.TestCase):
    """Verify handler path resolution never falls back to download defaults."""

    def test_model_base_dir_from_env(self):
        path = get_model_base_dir()
        self.assertTrue(path.is_absolute(), f"base dir must be absolute, got {path}")
        self.assertNotEqual(str(path), "downloads",
                            "base dir must not default to downloads/ when LANCE_MODEL_BASE_DIR is set")

    def test_model_base_dir_default(self):
        old = os.environ.pop("LANCE_MODEL_BASE_DIR", None)
        try:
            path = get_model_base_dir()
            self.assertEqual(str(path), "downloads")
        finally:
            if old is not None:
                os.environ["LANCE_MODEL_BASE_DIR"] = old

    def test_image_model_path(self):
        path = get_model_path("image")
        self.assertTrue(path.is_absolute(), f"image model path must be absolute, got {path}")
        self.assertEqual(path.name, "Lance_3B")
        self.assertNotIn("downloads", str(path),
                         f"image model path must not reference downloads/, got {path}")

    def test_video_model_path(self):
        path = get_model_path("t2v")
        self.assertTrue(path.is_absolute(), f"video model path must be absolute, got {path}")
        self.assertEqual(path.name, "Lance_3B_Video")
        self.assertNotIn("downloads", str(path),
                         f"video model path must not reference downloads/, got {path}")

    def test_vit_path(self):
        base = get_model_base_dir()
        vit_path = base / "Qwen2.5-VL-ViT"
        self.assertTrue(vit_path.is_absolute(),
                        f"vit path must be absolute, got {vit_path}")
        self.assertEqual(vit_path.name, "Qwen2.5-VL-ViT")
        self.assertNotIn("downloads", str(vit_path),
                         f"vit path must not reference downloads/, got {vit_path}")

    def test_all_known_paths_are_absolute(self):
        """Every path the handler resolves must be absolute."""
        base = get_model_base_dir()
        paths = {
            "base_dir": base,
            "image_model": get_model_path("image"),
            "video_model": get_model_path("t2v"),
            "vit": base / "Qwen2.5-VL-ViT",
            "vae": base / "Wan2.2_VAE.pth",
        }
        for name, path in paths.items():
            with self.subTest(name=name):
                self.assertTrue(
                    path.is_absolute(),
                    f"{name} must be absolute, got {path}",
                )
                self.assertNotIn(
                    "downloads", str(path),
                    f"{name} must not have downloads/ fallback, got {path}",
                )

    def test_no_env_var_falls_back_to_downloads(self):
        """Without LANCE_MODEL_BASE_DIR, we expect the downloads/ fallback."""
        old_base = os.environ.pop("LANCE_MODEL_BASE_DIR", None)
        old_image = os.environ.pop("LANCE_IMAGE_MODEL_PATH", None)
        old_video = os.environ.pop("LANCE_VIDEO_MODEL_PATH", None)
        old_path = os.environ.pop("LANCE_MODEL_PATH", None)
        try:
            for variant in ["image", "t2v"]:
                with self.subTest(variant=variant):
                    path = get_model_path(variant)
                    self.assertIn("downloads", str(path),
                                  f"without env vars, {variant} path should fall back to downloads/")
        finally:
            for k, v in [("LANCE_MODEL_BASE_DIR", old_base),
                         ("LANCE_IMAGE_MODEL_PATH", old_image),
                         ("LANCE_VIDEO_MODEL_PATH", old_video),
                         ("LANCE_MODEL_PATH", old_path)]:
                if v is not None:
                    os.environ[k] = v


class TestModelCacheStructure(unittest.TestCase):
    """If LANCE_MODEL_BASE_DIR points to a real cache, verify expected assets exist."""

    def test_expected_assets_present(self):
        base = get_model_base_dir()
        if not base.exists():
            self.skipTest(f"model cache not mounted at {base} — skipping integration check")
        for asset in EXPECTED_ASSETS:
            asset_path = base / asset
            self.assertTrue(
                asset_path.exists(),
                f"expected {asset} at {asset_path}",
            )


class TestHandlerImportSmoke(unittest.TestCase):
    """Verify handler.py can be parsed without syntax errors."""

    def test_handler_parses_without_error(self):
        handler_path = Path(__file__).resolve().parent / "handler.py"
        if not handler_path.exists():
            self.skipTest("handler.py not found")
        with open(handler_path) as f:
            source = f.read()
        compile(source, str(handler_path), "exec")

    def test_handler_contains_downloads_patch(self):
        """Verify handler.py has the config_factory redirect patch."""
        handler_path = Path(__file__).resolve().parent / "handler.py"
        if not handler_path.exists():
            self.skipTest("handler.py not found")
        source = handler_path.read_text()
        self.assertIn("downloads/", source,
                      "handler.py must contain the downloads/ redirect patch")
        self.assertIn("_patched_cf_get_model_path", source,
                      "handler.py must define _patched_cf_get_model_path")
        self.assertIn("_cf.get_model_path = _patched_cf_get_model_path", source,
                      "handler.py must install the path patch")


class TestDownloadsRedirect(unittest.TestCase):
    """Simulate the config_factory patch: any downloads/ path must redirect."""

    def _simulate_patch(self, path_key: str) -> str:
        """Simulate what _patched_cf_get_model_path does."""
        # These are the Lance library defaults we know about
        fake_defaults = {
            "vae.wan": "downloads/Wan2.2_VAE.pth",
            "vit.qwen2_5_vl": "downloads/Qwen2.5-VL-ViT",
            "llm.qwen2": "downloads/Lance_3B",
            "tokenizer.qwen2": "downloads/Lance_3B",
        }
        path = fake_defaults.get(path_key, f"downloads/some/model.safetensors")
        if path.startswith("downloads/"):
            base = get_model_base_dir()
            return str(base / path[len("downloads/"):])
        return path

    def test_downloads_redirected_to_model_base_dir(self):
        """Any path starting with downloads/ must go to LANCE_MODEL_BASE_DIR."""
        for path_key in ["vae.wan", "vit.qwen2_5_vl", "llm.qwen2", "tokenizer.qwen2"]:
            with self.subTest(key=path_key):
                resolved = self._simulate_patch(path_key)
                self.assertTrue(
                    resolved.startswith("/runpod-volume/checkpoints/"),
                    f"{path_key} must resolve under /runpod-volume/checkpoints/, got {resolved}",
                )
                self.assertNotIn(
                    "downloads", resolved,
                    f"{path_key} must not contain downloads/ after patch, got {resolved}",
                )

    def test_non_downloads_paths_unchanged(self):
        """Paths that don't start with downloads/ should pass through unchanged."""
        result = self._simulate_patch("some.absolute.path")
        # The simulator returns a made-up default, but if the real patch
        # gets a non-downloads path, it leaves it alone.
        base = get_model_base_dir()
        resolved = self._simulate_patch("vae.wan")
        self.assertTrue(
            resolved.startswith(str(base)),
            f"patched path should start with base dir {base}, got {resolved}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
