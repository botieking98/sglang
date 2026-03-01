# Unified JIT / Precompilation Cache Directory

**Issue:** https://github.com/sgl-project/sglang/issues/19612

**Author:** @whybeyoung
**Status:** Design Approved

---

## Summary

Unify all JIT and precompilation cache paths under a single configurable root so that users and operators can manage cache location, size, and persistence in one place.

---

## Problem Statement

Currently, SGLang uses fragmented cache locations for different JIT components:

| Component | Environment Variable | Default Path |
|-----------|---------------------|--------------|
| DeepGEMM | `SGLANG_DG_CACHE_DIR` / `DG_JIT_CACHE_DIR` | `~/.cache/deep_gemm` |
| Triton | `TRITON_CACHE_DIR` | `$SGLANG_CACHE_DIR/triton_cache` |
| Inductor | `TORCHINDUCTOR_CACHE_DIR` | `$SGLANG_CACHE_DIR/inductor_cache` |
| torch.compile | `SGLANG_CACHE_DIR` | `~/.cache/sglang/` |
| custom_all_reduce | (hardcoded) | `~/.cache/sglang` |

**Problems:**
1. **No single root:** Triton, Inductor, SGLang torch.compile, and DeepGEMM each have their own env or default
2. **Mixed storage:** Some write to `/tmp`, others to `~/.cache/...`
3. **Difficult management:** Hard to reason about compiled artifact locations or reuse caches across runs

---

## Design

### New Environment Variable

Introduce **`SGLANG_JIT_CACHE_ROOT`** as the unified cache root directory.

### Default Path

- Default: `~/.cache/sglang` (or `$XDG_CACHE_HOME/sglang` when set)
- Respects XDG Base Directory Specification

### Directory Layout

```
$SGLANG_JIT_CACHE_ROOT/              # Default: ~/.cache/sglang
├── triton/                          # → TRITON_CACHE_DIR
├── inductor/                        # → TORCHINDUCTOR_CACHE_DIR
├── torch_compile/                   # → torch.compile cache
│   └── <hash>/
│       └── rank_0_0/
│           └── <model_tag>/
├── deep_gemm/                       # → DG_JIT_CACHE_DIR
└── gpu_p2p_access_cache_*.json      # custom_all_reduce cache
```

### Backward Compatibility

Existing environment variables take precedence over the new unified scheme:

1. If specific env var is set (e.g., `SGLANG_TRITON_CACHE_DIR`), use it
2. Else if `SGLANG_JIT_CACHE_ROOT` is set, derive path as `{root}/{subdir}`
3. Else use the component's legacy default

### Component Mapping

| Component | Derived From | Subdirectory | Legacy Override |
|-----------|--------------|--------------|-----------------|
| Triton | `SGLANG_JIT_CACHE_ROOT` | `triton/` | `SGLANG_TRITON_CACHE_DIR` |
| Inductor | `SGLANG_JIT_CACHE_ROOT` | `inductor/` | `SGLANG_INDUCTOR_CACHE_DIR` |
| torch.compile | `SGLANG_JIT_CACHE_ROOT` | `torch_compile/` | `SGLANG_CACHE_DIR` |
| DeepGEMM | `SGLANG_JIT_CACHE_ROOT` | `deep_gemm/` | `SGLANG_DG_CACHE_DIR` |
| custom_all_reduce | `SGLANG_JIT_CACHE_ROOT` | (root) | N/A |

---

## Implementation Approach

### 1. Centralized Configuration Module

Create `sglang/srt/cache_config.py` to handle all cache path configuration:

- `configure_jit_cache_root()` : Called early in server initialization
- `get_jit_cache_root()` : Returns the effective cache root
- `get_cache_path(component)` : Returns the cache path for a specific component

### 2. Environment Variable Registration

Add `SGLANG_JIT_CACHE_ROOT` to `sglang/srt/environ.py`:

```python
SGLANG_JIT_CACHE_ROOT = EnvStr(None)  # None means use XDG default
```

### 3. Entry Point Integration

Call `configure_jit_cache_root()` early in:
- `sglang/srt/entrypoints/http_server.py`
- `sglang/srt/entrypoints/engine.py`
- `sglang/compile_deep_gemm.py`

### 4. Refactor Existing Code

Update the following files to use the centralized configuration:

- `sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`
- `sglang/srt/compilation/compiler_interface.py`
- `sglang/srt/compilation/backend.py`
- `sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py`

### 5. Testing Strategy

- Unit tests for `cache_config.py` functions
- Integration tests verifying all cache paths are correctly set
- Backward compatibility tests ensuring legacy env vars still work

---

## API Example

```bash
# Use unified cache root
export SGLANG_JIT_CACHE_ROOT=/shared/cache/sglang

# Or use specific overrides (takes precedence)
export SGLANG_TRITON_CACHE_DIR=/fast/local/triton_cache

# Launch server
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8
```

Cache structure after running:
```
/shared/cache/sglang/
├── triton/
├── inductor/
├── torch_compile/
│   └── abc123def4/
│       └── rank_0_0/
│           └── DeepSeek-V3/
└── deep_gemm/
```

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing cache setups | Backward compatibility: legacy env vars take precedence |
| Race condition during directory creation | Use `os.makedirs(exist_ok=True)` |
| Wrong cache root detected | Log the effective cache paths at INFO level |

---

## Future Work

- Cache size limiting and cleanup utilities
- Cache sharing utilities for multi-node setups
- Cache validation and migration tools
