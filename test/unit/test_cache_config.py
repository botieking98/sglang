"""Tests for sglang.srt.cache_config module."""

import os
import tempfile
import shutil
import unittest
from unittest.mock import patch

from sglang.srt.cache_config import (
    configure_jit_cache_root,
    get_jit_cache_root,
    get_cache_path,
    get_custom_all_reduce_cache_path,
    _get_default_cache_root,
)


class TestCacheConfig(unittest.TestCase):
    """Test cases for cache configuration."""

    def setUp(self):
        """Save and clear environment before each test."""
        self.saved_env = {}
        env_vars = [
            "SGLANG_JIT_CACHE_ROOT",
            "XDG_CACHE_HOME",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "SGLANG_CACHE_DIR",
            "DG_JIT_CACHE_DIR",
            "SGLANG_TRITON_CACHE_DIR",
            "SGLANG_INDUCTOR_CACHE_DIR",
            "SGLANG_TORCH_COMPILE_CACHE_DIR",
            "SGLANG_DG_CACHE_DIR",
        ]
        for var in env_vars:
            self.saved_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore environment after each test."""
        for var, value in self.saved_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_get_default_cache_root_xdg(self):
        """Test that XDG_CACHE_HOME is respected."""
        os.environ["XDG_CACHE_HOME"] = "/tmp/xdg_cache"
        root = _get_default_cache_root()
        self.assertEqual(root, "/tmp/xdg_cache/sglang")

    def test_get_default_cache_root_home(self):
        """Test fallback to home directory."""
        home = os.path.expanduser("~")
        root = _get_default_cache_root()
        self.assertEqual(root, os.path.join(home, ".cache", "sglang"))

    def test_get_jit_cache_root_env_var(self):
        """Test SGLANG_JIT_CACHE_ROOT takes precedence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SGLANG_JIT_CACHE_ROOT"] = tmpdir
            root = get_jit_cache_root()
            self.assertEqual(root, os.path.abspath(tmpdir))

    def test_get_jit_cache_root_expands_user(self):
        """Test that ~ is expanded in cache root."""
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "~/custom_cache"
        root = get_jit_cache_root()
        self.assertFalse(root.startswith("~"))
        self.assertTrue(root.endswith("custom_cache"))

    def test_get_cache_path_unknown_component(self):
        """Test unknown component returns None."""
        path = get_cache_path("unknown_component")
        self.assertIsNone(path)

    def test_get_cache_path_derived_from_root(self):
        """Test cache path derived from SGLANG_JIT_CACHE_ROOT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SGLANG_JIT_CACHE_ROOT"] = tmpdir
            path = get_cache_path("triton")
            self.assertEqual(path, os.path.join(tmpdir, "triton"))

    def test_get_cache_path_legacy_override(self):
        """Test legacy env var takes precedence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_triton")
            os.environ["SGLANG_JIT_CACHE_ROOT"] = tmpdir
            os.environ["SGLANG_TRITON_CACHE_DIR"] = custom_path
            path = get_cache_path("triton")
            self.assertEqual(path, custom_path)

    def test_get_cache_path_existing_target_env(self):
        """Test existing target env var is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "existing_triton")
            os.environ["TRITON_CACHE_DIR"] = custom_path
            path = get_cache_path("triton")
            self.assertEqual(path, custom_path)

    def test_get_cache_path_legacy_over_existing(self):
        """Test legacy env var takes precedence over existing target env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = os.path.join(tmpdir, "legacy_triton")
            existing_path = os.path.join(tmpdir, "existing_triton")
            os.environ["TRITON_CACHE_DIR"] = existing_path
            os.environ["SGLANG_TRITON_CACHE_DIR"] = legacy_path
            path = get_cache_path("triton")
            self.assertEqual(path, legacy_path)

    def test_configure_jit_cache_root_creates_dirs(self):
        """Test that configure_jit_cache_root creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "new_cache_root")
            os.environ["SGLANG_JIT_CACHE_ROOT"] = root
            result = configure_jit_cache_root()
            self.assertEqual(result, os.path.abspath(root))
            self.assertTrue(os.path.isdir(root))
            self.assertTrue(os.path.isdir(os.path.join(root, "triton")))
            self.assertTrue(os.path.isdir(os.path.join(root, "inductor")))
            self.assertTrue(os.path.isdir(os.path.join(root, "torch_compile")))
            self.assertTrue(os.path.isdir(os.path.join(root, "deep_gemm")))

    def test_configure_jit_cache_root_sets_env_vars(self):
        """Test that configure_jit_cache_root sets environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SGLANG_JIT_CACHE_ROOT"] = tmpdir
            configure_jit_cache_root()
            self.assertIn("TRITON_CACHE_DIR", os.environ)
            self.assertIn("TORCHINDUCTOR_CACHE_DIR", os.environ)
            self.assertIn("SGLANG_CACHE_DIR", os.environ)
            self.assertIn("DG_JIT_CACHE_DIR", os.environ)

    def test_configure_jit_cache_root_preserves_legacy(self):
        """Test that configure_jit_cache_root preserves legacy env vars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_deep_gemm")
            os.environ["SGLANG_DG_CACHE_DIR"] = custom_path
            os.environ["SGLANG_JIT_CACHE_ROOT"] = tmpdir
            configure_jit_cache_root()
            self.assertEqual(os.environ["DG_JIT_CACHE_DIR"], custom_path)

    def test_get_custom_all_reduce_cache_path(self):
        """Test custom_all_reduce cache path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SGLANG_JIT_CACHE_ROOT"] = tmpdir
            path = get_custom_all_reduce_cache_path("0,1,2,3")
            expected = os.path.join(tmpdir, "gpu_p2p_access_cache_for_0,1,2,3.json")
            self.assertEqual(path, expected)

    def test_all_components(self):
        """Test all known cache components."""
        components = ["triton", "inductor", "torch_compile", "deep_gemm"]
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SGLANG_JIT_CACHE_ROOT"] = tmpdir
            for component in components:
                path = get_cache_path(component)
                self.assertIsNotNone(path)
                self.assertTrue(path.startswith(os.path.abspath(tmpdir)))


if __name__ == "__main__":
    unittest.main()
