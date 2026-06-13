"""Policy server.

Two policy modes:
  policy:checkpoint  -- load JAX/orbax checkpoint via openpi (original behavior)
  policy:gguf        -- load quantized GGUF and run via OmniModel.cpp Python binding

GGUF mode example (put ``--port`` before the ``policy:`` subcommand):
    uv run scripts/serve_policy_gguf.py --port=8000 policy:gguf \
        --policy.dir=/path/to/pi05_bundle \
        --policy.device=CUDA0 --policy.steps=10

    Vision encoder runs on the same backend as ``--policy.device`` by default
    (e.g. CUDA0 → CUDA0). Override with ``--policy.vision-device=CPU`` if the
    GGUF is FP16 and the vision-backend compute scratch (which ggml-cuda over-
    allocates for FP16 SigLIP, ~12 GiB) is too big for your card.
"""
import dataclasses
import enum
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi_client import base_policy as _base_policy


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint (JAX/orbax)."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class GGUF:
    """Load a quantized GGUF model and run inference via OmniModel.cpp Python binding.

    Expected directory layout:
        <dir>/pi05.gguf
        <dir>/tokenizer.model
        <dir>/norm_stats.json
    """

    # Directory holding pi05.gguf + tokenizer.model + norm_stats.json
    dir: str
    # Device for ggml backend ("CUDA", "CPU", etc.)
    device: str = "CUDA"
    # CPU threads (only relevant for CPU device)
    n_threads: int = 4
    # Flow-matching ODE steps
    steps: int = 10
    # Optional: ODE step profiling output path
    ode_profile_path: str = ""
    # Path to the OmniModel.cpp build directory containing pi05.so (Python binding)
    binding_path: str = "/home/user1/arash/OmniModel.cpp/build_openpi/bin"
    # Action dimension exposed to the client (e.g. 7 for UR3, libero). The model's
    # padded action_dim (typically 32) is read from the Pi05Pipeline at load time.
    action_dim: int = 7
    # Where to load the vision encoder (v.* / mm.*). Empty string = same backend
    # as ``device`` (recommended — vision on the same GPU is ~30× faster than CPU
    # vision, and ggml's compute scratch is small enough on quantized checkpoints).
    # Set explicitly to ``CPU`` to save VRAM on FP16 deployments where ggml-cuda
    # over-allocates the SigLIP forward-pass scratch (~12 GiB extra for FP16).
    vision_device: str = ""
    # Optional default prompt (used if obs has no "prompt" key)
    default_prompt: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | GGUF | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


# ─── GGUF policy adapter ─────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def _load_norm_stats(model_dir: Path) -> dict | None:
    """Read action + state norm stats from norm_stats.json.

    Supports two schemas:
    - FLAT (what export_pi05.py writes): {"action_q01": [...], "action_q99": [...], "action_mean": [...], ...,
                                          "state_q01": [...], "state_q99": [...], "state_mean": [...], ...}
    - NESTED (some openpi assets):       {"action": {"q01": [...], ...}, "state": {"q01": [...], ...}}

    Pi 0.5 in OpenPI is trained with QUANTILE normalization (use_quantile_norm=True for
    model_type != PI0). We prefer q01/q99 when both are available; otherwise fall back to
    mean/std. Returns None only if no action stats are present.

    Output schema (flat per-key):
        out["action"] = {"use_quantile": bool, "q01"/"q99" or "mean"/"std": np.ndarray}
        out["state"]  = same, or missing if not in file
    """
    import json
    p = model_dir / "norm_stats.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    if "norm_stats" in data:
        data = data["norm_stats"]

    def _read_stats(key_prefix: str, nested_key: str) -> dict[str, Any] | None:
        nested = data.get(nested_key) or data.get(nested_key + "s") or {}
        q01  = data.get(f"{key_prefix}_q01")  if f"{key_prefix}_q01"  in data else nested.get("q01")
        q99  = data.get(f"{key_prefix}_q99")  if f"{key_prefix}_q99"  in data else nested.get("q99")
        mean = data.get(f"{key_prefix}_mean") if f"{key_prefix}_mean" in data else nested.get("mean")
        std  = data.get(f"{key_prefix}_std")  if f"{key_prefix}_std"  in data else nested.get("std")

        sub: dict[str, Any] = {}
        if q01 is not None and q99 is not None:
            sub["use_quantile"] = True
            sub["q01"] = np.asarray(q01, dtype=np.float32)
            sub["q99"] = np.asarray(q99, dtype=np.float32)
            return sub
        if mean is not None and std is not None:
            sub["use_quantile"] = False
            sub["mean"] = np.asarray(mean, dtype=np.float32)
            sub["std"]  = np.asarray(std,  dtype=np.float32)
            return sub
        return None

    out: dict[str, Any] = {}
    action_stats = _read_stats("action", "action")
    if action_stats is None:
        logger.warning("norm_stats.json present but no usable action stats — keys: %s", sorted(data.keys()))
        return None
    out["action"] = action_stats
    a_dim = action_stats.get("q01", action_stats.get("mean")).shape[0]
    logger.info("Loaded action norm stats (dim=%d, mode=%s) from %s",
                a_dim, "quantile" if action_stats["use_quantile"] else "mean_std", p)

    state_stats = _read_stats("state", "state")
    if state_stats is not None:
        out["state"] = state_stats
        s_dim = state_stats.get("q01", state_stats.get("mean")).shape[0]
        logger.info("Loaded state norm stats (dim=%d, mode=%s)",
                    s_dim, "quantile" if state_stats["use_quantile"] else "mean_std")
    else:
        logger.warning("No state norm stats found — model will receive zero state, "
                       "which usually breaks state-conditioned policies.")
    return out


def _normalize_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """OpenPI quantile normalization → roughly [-1, 1]."""
    x = np.asarray(x, dtype=np.float32)
    D = x.shape[-1]
    return 2.0 * (x - q01[:D]) / (q99[:D] - q01[:D] + 1e-6) - 1.0


def _unnormalize_actions(actions: np.ndarray, ns_root: dict) -> np.ndarray:
    """Reverse the normalization that pi 0.5 applied to the action distribution at training time."""
    ns = ns_root.get("action", ns_root) if "action" in ns_root or "use_quantile" not in ns_root else ns_root
    D = actions.shape[-1]
    if ns.get("use_quantile", False):
        q01, q99 = ns["q01"][:D], ns["q99"][:D]
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    if "mean" in ns:
        m, s = ns["mean"][:D], ns["std"][:D]
        return actions * s + m
    return actions


def _normalize_and_pad_state(state: np.ndarray, ns_root: dict, padded_dim: int) -> np.ndarray:
    """Normalize incoming proprio state to match training distribution, then zero-pad to ``padded_dim``.

    Uses the same scheme (quantile / mean_std) the training pipeline used. Returns an
    np.float32 vector of length ``padded_dim`` suitable for handing to the binding.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    ns = ns_root.get("state") if isinstance(ns_root, dict) else None
    if ns is None:
        # No norm stats — best we can do is pad raw state. This will almost certainly
        # be off-distribution; we emit a one-shot warning at load time.
        normed = state
    elif ns.get("use_quantile", False):
        normed = _normalize_quantile(state, ns["q01"], ns["q99"])
    else:
        m, s = ns["mean"], ns["std"]
        D = state.shape[-1]
        normed = (state - m[:D]) / np.maximum(s[:D], 1e-6)

    out = np.zeros(padded_dim, dtype=np.float32)
    n = min(normed.shape[-1], padded_dim)
    out[:n] = normed[:n]
    return out


# Common image-key aliases the policy will look for in the obs dict.
_BASE_IMAGE_KEYS = (
    "observation.images.fixed",       # UR3 / LeRobot style
    "observation.images.image",       # libero
    "observation/image",              # libero (alt)
    "image",
    "base_image",
)
_WRIST_IMAGE_KEYS = (
    "observation.images.cam_wrist",   # UR3 / LeRobot style
    "observation.images.wrist_image",
    "observation/wrist_image",
    "wrist_image",
)

# Common proprio-state aliases.
_STATE_KEYS = (
    "state",                          # LeRobot generic
    "observation.state",              # LeRobot dotted
    "observation/state",
    "robot_state",
)


def _pick_image(obs: dict, keys: tuple) -> np.ndarray | None:
    for k in keys:
        v = obs.get(k)
        if v is not None:
            return np.asarray(v)
    return None


def _pick_state(obs: dict) -> np.ndarray | None:
    """Return raw proprio state from common observation keys.

    Falls back to assembling joint_position + gripper_position when present (typical
    UR3 client format). Returns None only when nothing usable is found.
    """
    for k in _STATE_KEYS:
        v = obs.get(k)
        if v is not None:
            return np.asarray(v, dtype=np.float32).reshape(-1)
    jp = obs.get("joint_position")
    gp = obs.get("gripper_position")
    if jp is not None and gp is not None:
        jp = np.asarray(jp, dtype=np.float32).reshape(-1)
        gp = np.asarray(gp, dtype=np.float32).reshape(-1)
        return np.concatenate([jp, gp], axis=0)
    return None


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (img * 255.0).clip(0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def _resolved_vision_device(device: str, vision_device: str) -> str:
    """Pi05Pipeline vision_device_name: '' means same backend as main device.

    Default policy: vision follows ``device``. Earlier this function forced
    CPU when ``device`` was CUDA* (to dodge ggml-cuda's pessimistic SigLIP
    scratch allocation), but that traded ~30× slower vision encoding for a
    one-time VRAM saving. With the quantized GGUFs the saving is small and
    not worth the latency, so the default is now "same backend as device".
    Pass an explicit ``CPU`` (or ``CUDA1``, etc.) to override.
    """
    return vision_device  # empty → binding interprets as "same backend as device"


# ─── VRAM + weight-space telemetry ───────────────────────────────────────────

def _cuda_index_from_device(device: str) -> int | None:
    """Parse 'CUDA0' / 'CUDA1' → 0/1. Returns None for CPU."""
    d = (device or "").strip().upper()
    if not d.startswith("CUDA"):
        return None
    tail = d[len("CUDA"):]
    if not tail:
        return 0  # plain 'CUDA' → first visible device
    try:
        return int(tail)
    except ValueError:
        return None


class _VramSampler:
    """pynvml wrapper for THIS process's GPU memory usage on a single device.

    Reports own-process bytes (so a shared GPU's other users don't pollute
    the number). Falls back to None values when CUDA / pynvml isn't available
    (CPU device, or pynvml import failed).
    """

    _initialized = False

    def __init__(self, gpu_index: int | None):
        self.handle = None
        self.pid = os.getpid()
        if gpu_index is None:
            return
        try:
            import pynvml  # type: ignore
            if not _VramSampler._initialized:
                pynvml.nvmlInit()
                _VramSampler._initialized = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.pynvml = pynvml
        except Exception as e:
            logger.warning("VRAM sampling disabled (pynvml init failed: %s)", e)

    def sample_mb(self) -> float | None:
        if self.handle is None:
            return None
        try:
            procs = self.pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            return sum(p.usedGpuMemory for p in procs if p.pid == self.pid) / 1024**2
        except Exception:
            return None


def _gguf_weight_breakdown(model_path: Path) -> dict[str, Any]:
    """Parse a GGUF file and return total params + dtype/component byte breakdown.

    Returns:
        dict with keys: total_params, file_bytes, bpw, by_dtype_mb (dict),
                        by_component_mb (dict). All sizes in MB except file_bytes.
        Returns empty dict if gguf-py isn't importable.
    """
    try:
        import gguf  # type: ignore
        from collections import Counter, defaultdict
    except ImportError:
        return {}

    try:
        r = gguf.GGUFReader(str(model_path))
    except Exception as e:
        logger.warning("GGUF inspection failed (%s)", e)
        return {}

    by_dtype: Counter = Counter()
    by_component: dict = defaultdict(int)
    total_params = 0
    for t in r.tensors:
        by_dtype[t.tensor_type.name] += t.n_bytes
        by_component[t.name.split(".")[0]] += t.n_bytes
        total_params += int(np.prod(t.shape))

    fsize = model_path.stat().st_size
    return {
        "total_params": total_params,
        "file_bytes": fsize,
        "bpw": (fsize * 8) / total_params if total_params else 0.0,
        "by_dtype_mb": {k: v / 1024**2 for k, v in sorted(by_dtype.items(), key=lambda x: -x[1])},
        "by_component_mb": {k: v / 1024**2 for k, v in sorted(by_component.items(), key=lambda x: -x[1])},
    }


# Pi 0.5 architecture constants used to size the static KV cache slab the binding
# pre-allocates on the action-expert CUDA backend. From action_expert.hpp:
#   18 transformer layers × head_dim=256 × max_prefix=1024 × {K,V}=2 × sizeof(F32)=4
#   = 37,748,736 bytes ≈ 36 MB (printed at load: "KV cache allocated on GPU: ...").
_KV_CACHE_BYTES = 18 * 256 * 1024 * 2 * 4


def _on_cuda_components(device: str, vision_device: str) -> set[str]:
    """Which GGUF component prefixes end up on the CUDA backend at load time.

    Mirrors the C++ binding's backend split:
      - `action.*` and `pali.*` live on the action-expert backend (= `device`).
      - `v.*` and `mm.*` live on the vision backend (= `vision_device`,
        which defaults to `device` if empty).
      - `embed.*` is always on host (CPU) — text_embed.hpp dequantizes per-row
        from a `void*` buffer, never touching CUDA.
    """
    on_cuda = set()
    if device.upper().startswith("CUDA"):
        on_cuda.update({"action", "pali"})
    effective_vision = vision_device or device  # empty → same as device
    if effective_vision.upper().startswith("CUDA"):
        on_cuda.update({"v", "mm"})
    return on_cuda


def _format_weight_summary(info: dict) -> list[str]:
    if not info:
        return ["  (gguf-py not available — skipping weight-space summary)"]
    lines = []
    fsize_gb = info["file_bytes"] / 1024**3
    lines.append(f"  file size on disk : {fsize_gb:6.2f} GB")
    lines.append(f"  total parameters  : {info['total_params']/1e9:6.3f} B")
    lines.append(f"  effective bpw     : {info['bpw']:6.2f}  bits/param  (whole file, incl. F32 norms)")
    lines.append("  weights per dtype : " + "  ".join(
        f"{k}={v/1024:.2f}GB" for k, v in info["by_dtype_mb"].items()))
    lines.append("  weights per comp  : " + "  ".join(
        f"{k}={v/1024:.2f}GB" for k, v in info["by_component_mb"].items()))
    return lines


class Pi05GGUFPolicy(_base_policy.BasePolicy):
    """Drop-in policy that runs Pi 0.5 inference via the OmniModel.cpp Python binding.

    Implements the same interface as openpi.policies.policy.Policy:
      - infer(obs) -> {"actions": np.ndarray (T, D), "policy_timing": {...}}
      - .metadata
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "CUDA",
        n_threads: int = 4,
        num_flow_steps: int = 10,
        ode_profile_path: str = "",
        binding_path: str = "/home/user1/arash/OmniModel.cpp/build_openpi/bin",
        default_prompt: str | None = None,
        action_dim: int = 7,
        vision_device: str = "",
    ):
        if binding_path not in sys.path:
            sys.path.insert(0, binding_path)
        try:
            import pi05  # noqa: F401  (binding from OmniModel.cpp build dir)
        except ImportError as e:
            raise ImportError(
                f"Could not import 'pi05' binding from {binding_path}.\n"
                f"Make sure OmniModel.cpp was built with the Python binding (pi05.so) "
                f"and that the path is correct.\nOriginal error: {e}"
            )
        self._model_dir = Path(model_dir)
        self._default_prompt = default_prompt

        model_path = self._model_dir / "pi05.gguf"
        tokenizer_path = self._model_dir / "tokenizer.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Missing {tokenizer_path}")

        vision_dev = _resolved_vision_device(device, vision_device)

        # Print GGUF weight-space summary BEFORE loading (so size on disk + quant
        # breakdown is visible even if the load itself OOMs).
        weight_info = _gguf_weight_breakdown(model_path)
        logger.info("=== GGUF weight space ===")
        for line in _format_weight_summary(weight_info):
            logger.info(line)

        # VRAM sampler: tracks own-process GPU memory across load + infer.
        # gpu_index is parsed from the device string — when serving with
        # CUDA_VISIBLE_DEVICES=N, the process sees only one GPU as index 0.
        self._vram = _VramSampler(_cuda_index_from_device(device))
        vram_pre = self._vram.sample_mb()
        self._vram_peak_mb: float | None = None
        self._weight_load_mb: float | None = None

        logger.info(
            "Loading Pi 0.5 GGUF: %s (device=%s, vision_device=%s, steps=%d)",
            model_path,
            device,
            vision_dev or "(same as device)",
            num_flow_steps,
        )
        self._pipe = pi05.Pi05Pipeline(
            model_path=str(model_path),
            tokenizer_path=str(tokenizer_path),
            device_name=device,
            n_threads=n_threads,
            num_flow_steps=num_flow_steps,
            ode_profile_path=ode_profile_path,
            vision_device_name=vision_dev,
        )
        self._action_horizon = int(self._pipe.action_horizon)
        self._model_action_dim = int(self._pipe.action_dim)   # padded dim (e.g. 32)
        self._action_dim = action_dim                          # real dim (e.g. 7 for UR3)

        vram_post = self._vram.sample_mb()
        self._vram_post_load_mb: float | None = vram_post
        self._vram_pre_load_mb: float | None = vram_pre

        # Per-component "expected weights on CUDA" from the GGUF — for the
        # breakdown displayed below. Components routed to CPU (embed table,
        # and v.*/mm.* when vision_device=CPU) are excluded.
        cuda_components = _on_cuda_components(device, vision_dev)
        self._weights_on_cuda_mb: float | None = None
        self._kv_cache_mb: float = _KV_CACHE_BYTES / 1024**2
        if weight_info:
            self._weights_on_cuda_mb = sum(
                v for k, v in weight_info["by_component_mb"].items() if k in cuda_components
            )

        if vram_pre is not None and vram_post is not None:
            self._weight_load_mb = vram_post - vram_pre
            self._vram_peak_mb = vram_post
            logger.info("=== VRAM breakdown (after model load) ===")
            logger.info("  total VRAM:              %.2f GB", vram_post/1024)
            logger.info("  ├── CUDA baseline:       %.2f GB  (runtime, ggml ctx)", vram_pre/1024)
            if self._weights_on_cuda_mb is not None:
                compute_bufs = self._weight_load_mb - self._weights_on_cuda_mb - self._kv_cache_mb
                logger.info("  ├── model weights:       %.2f GB  (CUDA-resident: %s)",
                            self._weights_on_cuda_mb/1024,
                            "+".join(sorted(cuda_components)))
                logger.info("  ├── KV cache slab:       %.2f GB  (18 layers × 256 × 1024 × 2 × F32)",
                            self._kv_cache_mb/1024)
                logger.info("  └── compute buffers:     %.2f GB  (ggml graph activations pre-allocated)",
                            max(0.0, compute_bufs)/1024)
            else:
                logger.info("  └── total load delta:    %.2f GB  (no GGUF breakdown available)",
                            self._weight_load_mb/1024)

        self._norm_stats = _load_norm_stats(self._model_dir)
        if self._norm_stats is None:
            logger.warning(
                "No norm_stats.json found in %s — actions will be returned in raw model space",
                self._model_dir,
            )
        self._metadata = {
            "server_id": f"pi05_gguf_{int(time.time())}",
            "model_path": str(model_path),
            "vision_device": vision_dev or device,
            "device": device,
            "num_flow_steps": num_flow_steps,
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
        }
        logger.info("Pi 0.5 GGUF loaded — action_horizon=%d, action_dim=%d (model padded=%d)",
                    self._action_horizon, self._action_dim, self._model_action_dim)

    # ── BasePolicy API ────────────────────────────────────────────────────
    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        t0 = time.monotonic()
        base = _pick_image(obs, _BASE_IMAGE_KEYS)
        if base is None:
            raise ValueError(
                f"No base image found in observation. Keys present: {sorted(obs.keys())}. "
                f"Looked for any of: {_BASE_IMAGE_KEYS}"
            )
        wrist = _pick_image(obs, _WRIST_IMAGE_KEYS)
        prompt = obs.get("prompt") or self._default_prompt or ""
        base = _ensure_uint8(base)

        # Build the state token (already-normalized + zero-padded to the model's padded
        # action_dim; the binding feeds it into action_in_proj as suffix token 0).
        state_padded: np.ndarray | None = None
        raw_state = _pick_state(obs)
        if raw_state is not None and self._norm_stats is not None:
            state_padded = _normalize_and_pad_state(
                raw_state, self._norm_stats, self._model_action_dim
            )
        elif raw_state is not None:
            # No norm stats but we got raw state — still better than zero, but warn once.
            if not getattr(self, "_warned_state_no_stats", False):
                logger.warning(
                    "Got obs state but no norm_stats — feeding raw + padded state "
                    "(likely off-distribution)."
                )
                self._warned_state_no_stats = True
            state_padded = np.zeros(self._model_action_dim, dtype=np.float32)
            n = min(raw_state.shape[-1], self._model_action_dim)
            state_padded[:n] = raw_state[:n]
        else:
            if not getattr(self, "_warned_state_missing", False):
                logger.warning(
                    "No state found in obs (looked for %s and joint+gripper). "
                    "Falling back to zero state — this is what caused the C++ binding "
                    "to behave badly before; make sure the client sends a 'state' field.",
                    _STATE_KEYS,
                )
                self._warned_state_missing = True

        if wrist is not None and wrist.size > 0:
            wrist = _ensure_uint8(wrist)
            actions_flat = self._pipe.run_multi(base, wrist, prompt, state=state_padded)
        else:
            actions_flat = self._pipe.run(base, prompt, state=state_padded)

        # C++ output is flat (action_horizon * model_action_dim,) — reshape and slice.
        T, D_padded = self._action_horizon, self._model_action_dim
        expected = T * D_padded
        if actions_flat.size >= expected:
            acts = actions_flat[:expected].reshape(T, D_padded)[:, : self._action_dim]
        else:
            # Defensive fallback
            acts = actions_flat.reshape(T, -1)
            if acts.shape[1] > self._action_dim:
                acts = acts[:, : self._action_dim]
            elif acts.shape[1] < self._action_dim:
                pad = np.zeros((T, self._action_dim - acts.shape[1]), np.float32)
                acts = np.concatenate([acts, pad], axis=1)

        # Unnormalize using q01/q99 (or mean/std) from norm_stats.json.
        if self._norm_stats is not None:
            acts = _unnormalize_actions(acts.astype(np.float32), self._norm_stats)

        infer_ms = (time.monotonic() - t0) * 1000.0

        # Sample VRAM post-inference and update the running peak. The "scratch"
        # number is the activation + KV / compute-graph overhead above the
        # resident weight slab.
        timing: dict[str, float] = {"infer_ms": infer_ms}
        vram_now = self._vram.sample_mb()
        if vram_now is not None:
            self._vram_peak_mb = (
                vram_now if self._vram_peak_mb is None else max(self._vram_peak_mb, vram_now)
            )
            timing["vram_now_gb"] = vram_now / 1024
            timing["vram_peak_gb"] = self._vram_peak_mb / 1024

            if self._vram_post_load_mb is not None:
                # Activation/compute scratch sits ABOVE the static load slab.
                # The load slab itself already contains weights + KV cache + load overhead.
                activations = self._vram_peak_mb - self._vram_post_load_mb
                timing["vram_activations_peak_gb"] = activations / 1024

                # Full breakdown for the timing payload — split the load delta
                # into its sub-components so the client can see where bytes go.
                if self._weights_on_cuda_mb is not None and self._vram_pre_load_mb is not None:
                    weights_gb = self._weights_on_cuda_mb / 1024
                    baseline_gb = self._vram_pre_load_mb / 1024
                    kv_gb = self._kv_cache_mb / 1024
                    compute_bufs = max(
                        0.0,
                        (self._weight_load_mb or 0.0) - self._weights_on_cuda_mb - self._kv_cache_mb,
                    )
                    compute_bufs_gb = compute_bufs / 1024
                    timing.update({
                        "vram_baseline_gb": baseline_gb,
                        "vram_weights_gb": weights_gb,
                        "vram_kv_cache_gb": kv_gb,
                        "vram_compute_buffers_gb": compute_bufs_gb,
                    })
                    logger.info(
                        "infer: %.0f ms  |  peak=%.2f GB  "
                        "= baseline %.2f + weights %.2f + kv_cache %.2f + compute_bufs %.2f "
                        "+ activations %.2f",
                        infer_ms, self._vram_peak_mb/1024,
                        baseline_gb, weights_gb, kv_gb, compute_bufs_gb, activations/1024,
                    )
                else:
                    logger.info(
                        "infer: %.0f ms  |  peak=%.2f GB  weights+overhead=%.2f GB  "
                        "activations=%.2f GB",
                        infer_ms, self._vram_peak_mb/1024,
                        (self._weight_load_mb or 0.0)/1024, activations/1024,
                    )
            else:
                logger.info("infer: %.0f ms  |  VRAM now=%.2f GB  peak=%.2f GB",
                            infer_ms, vram_now/1024, self._vram_peak_mb/1024)

        return {
            "actions": acts.astype(np.float32),
            "policy_timing": timing,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


# ─── Policy creation dispatch ────────────────────────────────────────────────


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args):
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case GGUF():
            return Pi05GGUFPolicy(
                model_dir=args.policy.dir,
                device=args.policy.device,
                n_threads=args.policy.n_threads,
                num_flow_steps=args.policy.steps,
                ode_profile_path=args.policy.ode_profile_path,
                binding_path=args.policy.binding_path,
                action_dim=args.policy.action_dim,
                default_prompt=args.policy.default_prompt or args.default_prompt,
                vision_device=args.policy.vision_device,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
