# Unified JIT Cache Directory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement unified JIT and precompilation cache directory (`SGLANG_JIT_CACHE_ROOT`) to consolidate all cache paths under a single configurable root.

**Architecture:** Create a centralized `cache_config.py` module that configures all cache paths at server startup. The module respects XDG Base Directory Specification and maintains backward compatibility with existing environment variables.

**Tech Stack:** Python, pytest, SGLang SRT runtime

---

## Task 1: Read Existing Files

**Files:**
- Read: `python/sglang/srt/environ.py` (understand EnvStr pattern)
- Read: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` (current cache setup)
- Read: `python/sglang/srt/compilation/compiler_interface.py` (inductor/triton cache)
- Read: `python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py` (hardcoded cache)

**Context Gathering:**
Understand how existing environment variables are defined and how caches are currently configured.

---

## Task 2: Create cache_config.py Module

**Files:**
- Create: `python/sglang/srt/cache_config.py`

**Step 1: Write the module with core functions**

```python
"""Unified JIT cache configuration for SGLang.

This module centralizes all JIT and precompilation cache path configuration.
It supports the SGLANG_JIT_CACHE_ROOT environment variable for unified cache
management while maintaining backward compatibility with legacy env vars.
"""

import os
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# Cache component definitions: (env_var, subdirectory, legacy_env_var)
_CACHE_COMPONENTS: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {
    "triton": ("TRITON_CACHE_DIR", "triton", "SGLANG_TRITON_CACHE_DIR"),
    "inductor": ("TORCHINDUCTOR_CACHE_DIR", "inductor", "SGLANG_INDUCTOR_CACHE_DIR"),
    "torch_compile": ("SGLANG_CACHE_DIR", "torch_compile", "SGLANG_TORCH_COMPILE_CACHE_DIR"),
    "deep_gemm": ("DG_JIT_CACHE_DIR", "deep_gemm", "SGLANG_DG_CACHE_DIR"),
}


def _get_default_cache_root() -> str:
    """Get the default cache root directory.

    Respects XDG_CACHE_HOME environment variable, falling back to ~/.cache/sglang
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return os.path.join(xdg_cache, "sglang")
    return os.path.expanduser("~/.cache/sglang")


def get_jit_cache_root() -> str:
    """Get the effective JIT cache root directory.

    Priority:
    1. SGLANG_JIT_CACHE_ROOT environment variable
    2. XDG_CACHE_HOME/sglang
    3. ~/.cache/sglang

    Returns:
        The absolute path to the cache root directory.
    """
    root = os.environ.get("SGLANG_JIT_CACHE_ROOT")
    if root:
        return os.path.abspath(os.path.expanduser(root))
    return os.path.abspath(_get_default_cache_root())


def get_cache_path(component: str) -> Optional[str]:
    """Get the cache path for a specific component.

    Args:
        component: One of "triton", "inductor", "torch_compile", "deep_gemm"

    Returns:
        The absolute path to the component's cache directory, or None if
        component is unknown.
    """
    if component not in _CACHE_COMPONENTS:
        logger.warning(f"Unknown cache component: {component}")
        return None

    target_env, subdir, legacy_env = _CACHE_COMPONENTS[component]

    # Priority 1: Legacy environment variable (backward compatibility)
    if legacy_env and legacy_env in os.environ:
        path = os.environ[legacy_env]
        logger.debug(f"Using legacy env var {legacy_env} for {component}: {path}")
        return os.path.abspath(os.path.expanduser(path))

    # Priority 2: Already set target environment variable
    if target_env in os.environ:
        path = os.environ[target_env]
        logger.debug(f"Using existing {target_env} for {component}: {path}")
        return os.path.abspath(os.path.expanduser(path))

    # Priority 3: Derive from SGLANG_JIT_CACHE_ROOT
    root = get_jit_cache_root()
    if subdir:
        path = os.path.join(root, subdir)
    else:
        path = root

    logger.debug(f"Derived {component} cache path from root: {path}")
    return path


def configure_jit_cache_root() -> str:
    """Configure all JIT cache environment variables.

    This function should be called early in the SGLang server initialization,
    before any JIT compilation occurs.

    It sets up environment variables for:
    - TRITON_CACHE_DIR
    - TORCHINDUCTOR_CACHE_DIR
    - SGLANG_CACHE_DIR (for torch.compile)
    - DG_JIT_CACHE_DIR (for DeepGEMM)

    Returns:
        The effective cache root directory.
    """
    root = get_jit_cache_root()

    # Ensure root directory exists
    os.makedirs(root, exist_ok=True)

    configured = []

    for component in _CACHE_COMPONENTS:
        target_env, _, _ = _CACHE_COMPONENTS[component]
        path = get_cache_path(component)

        if path:
            os.makedirs(path, exist_ok=True)
            # Only set if not already configured by legacy env var
            if target_env not in os.environ:
                os.environ[target_env] = path
                configured.append(f"{target_env}={path}")

    if configured:
        logger.info(f"Configured JIT cache root: {root}")
        for cfg in configured:
            logger.debug(f"  {cfg}")

    return root


def get_custom_all_reduce_cache_path(cuda_visible_devices: str) -> str:
    """Get the cache path for custom_all_reduce GPU P2P access cache.

    Args:
        cuda_visible_devices: The CUDA_VISIBLE_DEVICES string

    Returns:
        Path to the GPU P2P access cache file.
    """
    root = get_jit_cache_root()
    filename = f"gpu_p2p_access_cache_for_{cuda_visible_devices}.json"
    return os.path.join(root, filename)
```

**Step 2: Verify file creation**

Run: `head -50 python/sglang/srt/cache_config.py`
Expected: Shows the module docstring and imports

**Step 3: Commit**

```bash
git add python/sglang/srt/cache_config.py
git commit -m "feat: add unified JIT cache configuration module

Add cache_config.py to centralize all JIT and precompilation cache
path configuration. Supports SGLANG_JIT_CACHE_ROOT env var and
maintains backward compatibility with legacy env vars."
```

---

## Task 3: Add Environment Variable to environ.py

**Files:**
- Modify: `python/sglang/srt/environ.py` (after line 355, near other cache-related vars)

**Step 1: Add the SGLANG_JIT_CACHE_ROOT environment variable**

Add after `SGLANG_DG_CACHE_DIR` line (around line 356):

```python
    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)

    # Unified JIT Cache Root (see cache_config.py)
    SGLANG_JIT_CACHE_ROOT = EnvStr(None)  # None means use XDG default
```

**Step 2: Verify the change**

Run: `grep -n "SGLANG_JIT_CACHE_ROOT" python/sglang/srt/environ.py`
Expected: Shows the new env var definition

**Step 3: Commit**

```bash
git add python/sglang/srt/environ.py
git commit -m "feat: add SGLANG_JIT_CACHE_ROOT environment variable

Add unified cache root env var to environ.py for configuring
all JIT and precompilation cache paths in one place."
```

---

## Task 4: Write Tests for cache_config.py

**Files:**
- Create: `test/unittest/test_cache_config.py`

**Step 1: Write comprehensive tests**

```python
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
```

**Step 2: Run tests (should pass)**

Run: `cd /Users/botieking/code/sglang && python -m pytest test/unittest/test_cache_config.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add test/unittest/test_cache_config.py
git commit -m "test: add comprehensive tests for cache_config module

Add unit tests covering:
- XDG_CACHE_HOME and default path resolution
- SGLANG_JIT_CACHE_ROOT precedence
- Legacy env var backward compatibility
- Directory creation
- All cache components"
```

---

## Task 5: Integrate into HTTP Server Entry Point

**Files:**
- Modify: `python/sglang/srt/entrypoints/http_server.py` (at the top of main entry point)

**Step 1: Find the import section and main function**

Look for imports (around line 1-50) and the main server startup function.

**Step 2: Add import at the top of the file**

After existing imports, add:

```python
from sglang.srt.cache_config import configure_jit_cache_root
```

**Step 3: Call configure_jit_cache_root early in the entry point**

Find where the server_args is parsed (usually near the main function). Add the call right after imports are processed but before any JIT compilation happens:

Around line where `ServerArgs` is used (search for `server_args =` or similar), add:

```python
    # Configure unified JIT cache before any compilation
    cache_root = configure_jit_cache_root()
    logger.info(f"SGLang JIT cache configured at: {cache_root}")
```

The exact location depends on the structure, but it should be early in the server initialization flow, ideally right after argument parsing.

**Step 4: Verify the change**

Run: `grep -n "configure_jit_cache_root\|cache_config" python/sglang/srt/entrypoints/http_server.py`
Expected: Shows both the import and the function call

**Step 5: Commit**

```bash
git add python/sglang/srt/entrypoints/http_server.py
git commit -m "feat: integrate unified JIT cache into HTTP server entry point

Call configure_jit_cache_root() early in server startup to ensure
all JIT caches are configured before any compilation occurs."
```

---

## Task 6: Integrate into Engine Entry Point

**Files:**
- Modify: `python/sglang/srt/entrypoints/engine.py` (early initialization)

**Step 1: Add import**

Add after existing imports:

```python
from sglang.srt.cache_config import configure_jit_cache_root
```

**Step 2: Call configure_jit_cache_root in engine initialization**

Find the `__init__` method or entry point of the Engine class and add early in initialization:

```python
        # Configure unified JIT cache
        cache_root = configure_jit_cache_root()
        logger.info(f"SGLang JIT cache configured at: {cache_root}")
```

**Step 3: Verify and commit**

Run: `grep -n "configure_jit_cache_root\|cache_config" python/sglang/srt/entrypoints/engine.py`
Expected: Shows both the import and the function call

```bash
git add python/sglang/srt/entrypoints/engine.py
git commit -m "feat: integrate unified JIT cache into engine entry point

Call configure_jit_cache_root() during engine initialization to
ensure all JIT caches are configured before any compilation."
```

---

## Task 7: Update DeepGEMM compile_utils.py

**Files:**
- Modify: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`

**Step 1: Replace hardcoded cache setup**

Remove lines 32-35:
```python
# OLD CODE (to be removed):
# Force redirect deep_gemm cache_dir
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGLANG_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)
```

Replace with:
```python
# Cache directory is now configured by cache_config.configure_jit_cache_root()
# which is called early in server/engine initialization.
# This import ensures we can access the cache path if configured.
from sglang.srt.cache_config import get_cache_path

# For backward compatibility, ensure DG_JIT_CACHE_DIR is set
# configure_jit_cache_root() should have been called by now
if "DG_JIT_CACHE_DIR" not in os.environ:
    cache_path = get_cache_path("deep_gemm")
    if cache_path:
        os.environ["DG_JIT_CACHE_DIR"] = cache_path
```

**Step 2: Add import at the top of the file**

Add after existing imports:
```python
from sglang.srt.cache_config import get_cache_path
```

**Step 3: Verify the change**

Run: `grep -A 5 "DG_JIT_CACHE_DIR" python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py | head -20`
Expected: Shows the new conditional code

**Step 4: Commit**

```bash
git add python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py
git commit -m "refactor: use unified cache config in DeepGEMM

Replace hardcoded cache directory setup with cache_config module.
DG_JIT_CACHE_DIR is now configured centrally at server startup."
```

---

## Task 8: Update compiler_interface.py

**Files:**
- Modify: `python/sglang/srt/compilation/compiler_interface.py`

**Step 1: Modify the initialize_cache method**

Current code (around line 190-195):
```python
# OLD CODE:
inductor_cache = os.path.join(self.base_cache_dir, "inductor_cache")
os.makedirs(inductor_cache, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
triton_cache = os.path.join(self.base_cache_dir, "triton_cache")
os.makedirs(triton_cache, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = triton_cache
```

Replace with:
```python
# Use cache_config if available, otherwise fallback to default behavior
from sglang.srt.cache_config import get_cache_path

# Get paths from centralized config or fallback to legacy behavior
inductor_cache = get_cache_path("inductor")
if inductor_cache is None:
    inductor_cache = os.path.join(self.base_cache_dir, "inductor_cache")
    os.makedirs(inductor_cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
elif "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache

triton_cache = get_cache_path("triton")
if triton_cache is None:
    triton_cache = os.path.join(self.base_cache_dir, "triton_cache")
    os.makedirs(triton_cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache
elif "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = triton_cache
```

**Step 2: Verify the change**

Run: `grep -A 10 "inductor_cache =" python/sglang/srt/compilation/compiler_interface.py | head -20`
Expected: Shows the new code using get_cache_path

**Step 3: Commit**

```bash
git add python/sglang/srt/compilation/compiler_interface.py
git commit -m "refactor: use unified cache config in compiler_interface

Update initialize_cache to use cache_config.get_cache_path() for
inductor and triton cache paths while maintaining backward compatibility."
```

---

## Task 9: Update custom_all_reduce_utils.py

**Files:**
- Modify: `python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py`

**Step 1: Find the hardcoded cache path**

Current code (around line 260-266):
```python
# OLD CODE:
# VLLM_CACHE_ROOT -> SGLANG_CACHE_ROOT
# "~/.cache/vllm" -> "~/.cache/sglang"
SGLANG_CACHE_ROOT = os.path.expanduser("~/.cache/sglang")
path = os.path.join(
    SGLANG_CACHE_ROOT, f"gpu_p2p_access_cache_for_{cuda_visible_devices}.json"
)
```

**Step 2: Replace with cache_config usage**

Replace the above with:
```python
# Use unified cache configuration
from sglang.srt.cache_config import get_custom_all_reduce_cache_path, get_jit_cache_root

path = get_custom_all_reduce_cache_path(cuda_visible_devices)
SGLANG_CACHE_ROOT = get_jit_cache_root()
```

**Step 3: Verify the change**

Run: `grep -A 5 "gpu_p2p_access_cache" python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py`
Expected: Shows the new code using get_custom_all_reduce_cache_path

**Step 4: Commit**

```bash
git add python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py
git commit -m "refactor: use unified cache config in custom_all_reduce

Replace hardcoded cache path with cache_config.get_custom_all_reduce_cache_path()
to respect SGLANG_JIT_CACHE_ROOT configuration."
```

---

## Task 10: Update compile_deep_gemm entry point

**Files:**
- Modify: `python/sglang/compile_deep_gemm.py` (entry point file)

**Step 1: Find the main function**

Look for the main entry point where the script starts.

**Step 2: Add early cache configuration**

Add at the beginning of main() or the entry point:

```python
from sglang.srt.cache_config import configure_jit_cache_root

# Configure unified JIT cache before any DeepGEMM operations
cache_root = configure_jit_cache_root()
logger.info(f"DeepGEMM compilation cache configured at: {cache_root}")
```

**Step 3: Verify and commit**

Run: `grep -n "configure_jit_cache_root" python/sglang/compile_deep_gemm.py`
Expected: Shows the function call

```bash
git add python/sglang/compile_deep_gemm.py
git commit -m "feat: add unified cache config to compile_deep_gemm entry point

Call configure_jit_cache_root() early to ensure DeepGEMM cache is
configured correctly even when running standalone deep_gemm compilation."
```

---

## Task 11: Update backend.py

**Files:**
- Modify: `python/sglang/srt/compilation/backend.py`

**Step 1: Find the SGLANG_CACHE_DIR usage**

Current code (around line 398-400):
```python
base_cache_dir = os.path.expanduser(
    os.getenv("SGLANG_CACHE_DIR", "~/.cache/sglang/")
)
```

**Step 2: Replace with cache_config usage**

Replace with:
```python
from sglang.srt.cache_config import get_cache_path

# Use unified cache config if available, otherwise fall back to env var
cache_path = get_cache_path("torch_compile")
if cache_path:
    base_cache_dir = cache_path
else:
    base_cache_dir = os.path.expanduser(
        os.getenv("SGLANG_CACHE_DIR", "~/.cache/sglang/")
    )
```

**Step 3: Verify and commit**

Run: `grep -n "get_cache_path\|base_cache_dir" python/sglang/srt/compilation/backend.py | head -10`
Expected: Shows the updated code

```bash
git add python/sglang/srt/compilation/backend.py
git commit -m "refactor: use unified cache config in compilation backend

Update backend.py to use cache_config.get_cache_path() for torch_compile
cache directory while maintaining backward compatibility."
```

---

## Task 12: Run All Unit Tests

**Files:**
- Run tests in: `test/unittest/test_cache_config.py`

**Step 1: Run the full test suite**

Run: `cd /Users/botieking/code/sglang && python -m pytest test/unittest/test_cache_config.py -v`
Expected: All tests pass

**Step 2: Test import doesn't break anything**

Run: `cd /Users/botieking/code/sglang && python -c "from sglang.srt.cache_config import configure_jit_cache_root; print('Import OK')"`
Expected: Prints "Import OK" without errors

**Step 3: Commit if all tests pass**

```bash
git status  # Verify all changes are committed
git log --oneline -5  # Show recent commits
```

---

## Task 13: Integration Testing

**Files:**
- Test file: Temporary test script

**Step 1: Create a simple integration test script**

```python
# test_integration.py - temporary integration test
import os
import sys
import tempfile
import shutil

# Clean environment
for key in list(os.environ.keys()):
    if 'CACHE' in key or 'JIT' in key:
        del os.environ[key]

# Set test cache root
test_root = tempfile.mkdtemp(prefix="sglang_test_cache_")
os.environ["SGLANG_JIT_CACHE_ROOT"] = test_root

print(f"Test cache root: {test_root}")

try:
    # Import and configure
    from sglang.srt.cache_config import configure_jit_cache_root, get_jit_cache_root

    root = configure_jit_cache_root()
    print(f"Configured root: {root}")

    # Verify all directories exist
    expected_dirs = ["triton", "inductor", "torch_compile", "deep_gemm"]
    for d in expected_dirs:
        path = os.path.join(root, d)
        if os.path.isdir(path):
            print(f"  ✓ {d}/ exists")
        else:
            print(f"  ✗ {d}/ MISSING")
            sys.exit(1)

    # Verify env vars are set
    env_vars = ["TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                "SGLANG_CACHE_DIR", "DG_JIT_CACHE_DIR"]
    for var in env_vars:
        if var in os.environ:
            print(f"  ✓ {var} = {os.environ[var]}")
        else:
            print(f"  ✗ {var} NOT SET")
            sys.exit(1)

    print("\n✓ Integration test PASSED")

finally:
    # Cleanup
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
        print(f"\nCleaned up: {test_root}")
```

**Step 2: Run the integration test**

Run: `cd /Users/botieking/code/sglang && python test_integration.py`
Expected: All checks pass

**Step 3: Clean up test file**

Run: `rm test_integration.py`

**Step 4: Commit**

```bash
git status
git log --oneline -10
```

---

## Task 14: Documentation Update

**Files:**
- Modify: `docs/references/deepseek.md` (or appropriate env var documentation file)

**Step 1: Find where environment variables are documented**

Look for existing env var documentation files or sections.

**Step 2: Add documentation for SGLANG_JIT_CACHE_ROOT**

Add to the environment variables section:

```markdown
### SGLANG_JIT_CACHE_ROOT

Unified root directory for all JIT and precompilation caches.

**Default:** `~/.cache/sglang` (or `$XDG_CACHE_HOME/sglang` if set)

This environment variable provides a single configuration point for all cache
paths used by SGLang's JIT compilation systems:

- `$SGLANG_JIT_CACHE_ROOT/triton/` → Triton cache
- `$SGLANG_JIT_CACHE_ROOT/inductor/` → PyTorch Inductor cache
- `$SGLANG_JIT_CACHE_ROOT/torch_compile/` → torch.compile cache
- `$SGLANG_JIT_CACHE_ROOT/deep_gemm/` → DeepGEMM JIT cache

**Backward Compatibility:**
- Legacy env vars (`SGLANG_TRITON_CACHE_DIR`, `SGLANG_DG_CACHE_DIR`, etc.) take precedence
- If a specific cache directory is already configured, `SGLANG_JIT_CACHE_ROOT` is ignored for that component

**Example:**
```bash
export SGLANG_JIT_CACHE_ROOT=/shared/fast_ssd/sglang_cache
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8
```
```

**Step 3: Verify documentation**

Review the added documentation for clarity and accuracy.

**Step 4: Commit**

```bash
git add docs/references/deepseek.md  # or appropriate file
git commit -m "docs: add SGLANG_JIT_CACHE_ROOT documentation

Document the unified JIT cache configuration including:
- Default path and XDG compliance
- Directory layout
- Backward compatibility behavior
- Usage example"
```

---

## Task 15: Final Verification

**Step 1: Full test run**

Run: `cd /Users/botieking/code/sglang && python -m pytest test/unittest/test_cache_config.py -v --tb=short`
Expected: All tests pass

**Step 2: Verify imports work in different contexts**

Run: `python -c "from sglang.srt.cache_config import configure_jit_cache_root; c = configure_jit_cache_root(); print(f'Cache root: {c}')"`
Expected: Shows the cache root path without errors

**Step 3: Review all changes**

Run: `git diff --stat HEAD~15` (adjust number to see all your commits)
Expected: Shows all modified and new files

**Step 4: Final commit if everything looks good**

```bash
git log --oneline -15
git status  # Should be clean
```

---

## Summary of Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `python/sglang/srt/cache_config.py` | Create | Centralized cache configuration module |
| `python/sglang/srt/environ.py` | Modify | Add `SGLANG_JIT_CACHE_ROOT` env var |
| `test/unittest/test_cache_config.py` | Create | Comprehensive unit tests |
| `python/sglang/srt/entrypoints/http_server.py` | Modify | Integrate cache config at startup |
| `python/sglang/srt/entrypoints/engine.py` | Modify | Integrate cache config at startup |
| `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` | Modify | Use cache_config for DG cache |
| `python/sglang/srt/compilation/compiler_interface.py` | Modify | Use cache_config for inductor/triton |
| `python/sglang/srt/compilation/backend.py` | Modify | Use cache_config for torch.compile |
| `python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py` | Modify | Use cache_config for P2P cache |
| `python/sglang/compile_deep_gemm.py` | Modify | Integrate cache config |
| Documentation file | Modify | Add SGLANG_JIT_CACHE_ROOT docs |

---

## Notes for Implementation

1. **Import Order Matters:** The `configure_jit_cache_root()` function must be called
   before any module that depends on cache environment variables is imported.

2. **Backward Compatibility:** The implementation prioritizes legacy environment variables
   to avoid breaking existing deployments.

3. **Idempotency:** Calling `configure_jit_cache_root()` multiple times is safe - it won't
   overwrite already-set environment variables.

4. **Test Coverage:** The unit tests cover normal operation, XDG compliance, backward
   compatibility, and edge cases.
