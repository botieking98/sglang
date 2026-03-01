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
