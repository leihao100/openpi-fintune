import dataclasses
import datetime
import enum
import io
import json
import logging
import pathlib
import socket

import flax.traverse_util
import matplotlib
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import image_tools
from PIL import Image
import tyro
from typing_extensions import override

matplotlib.use("Agg")  # headless: never open a window, just write PNGs
import matplotlib.pyplot as plt  # noqa: E402

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
import openpi.transforms as transforms


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


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

    # Number of flow-matching denoising steps used at inference time. More steps = more accurate ODE
    # integration (smoother/closer actions) but slower inference. Model default is 10.
    num_steps: int = 10

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Directory to log each inference's obs and predicted action chunk into.
    log_dir: str = "inference_logs"

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


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


def _is_image(arr: np.ndarray) -> bool:
    """Heuristic: an HxWx3 / HxWx1 / HxW array large enough to be an image."""
    return arr.ndim in (2, 3) and arr.shape[0] >= 16 and arr.shape[1] >= 16 and (
        arr.ndim == 2 or arr.shape[-1] in (1, 3)
    )


def _save_image(arr: np.ndarray, path: pathlib.Path) -> None:
    """Save an array as a PNG, normalizing common float ranges to uint8."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo = arr.min()
        if lo < 0.0:  # assume [-1, 1]
            arr = (arr + 1.0) / 2.0
        elif arr.max() > 1.0:  # assume [0, 255]
            arr = arr / 255.0
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


# dataviz palette: fixed categorical order (never cycled), single-series blue.
_SERIES_BLUE = "#2a78d6"
_GRID_GRAY = "#d8d7d2"


def _grid(n: int) -> tuple[int, int]:
    """Rows x cols for `n` small-multiple panels: at most 4 columns, near-square."""
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    return rows, cols


def _render_chunk(actions: np.ndarray, path: pathlib.Path, title: str) -> None:
    """Heatmap of one predicted action chunk: y = action dim, x = horizon step.

    A diverging colormap centered at 0 makes sign and magnitude readable at a
    glance, and it scales to any action dim (unlike one colored line per dim)."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[:, None]
    horizon, dim = actions.shape
    vmax = float(np.abs(actions).max()) or 1.0

    fig, ax = plt.subplots(figsize=(max(6.0, horizon * 0.16), max(2.4, dim * 0.4)))
    im = ax.imshow(
        actions.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("horizon step (predicted)")
    ax.set_ylabel("action dim")
    ax.set_yticks(range(dim))
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="action value")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _render_trajectory(
    history: np.ndarray, path: pathlib.Path, ylabel: str, title: str
) -> None:
    """Small-multiples line chart: one panel per dim, value over inference steps.

    Each panel is a single series, so it needs no legend and no color cycling —
    identity comes from the panel title, magnitude from the shared single hue."""
    history = np.asarray(history, dtype=np.float32)
    if history.ndim == 1:
        history = history[:, None]
    steps, dim = history.shape
    rows, cols = _grid(dim)
    x = np.arange(steps)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3.0, rows * 1.7), sharex=True, squeeze=False,
    )
    for d in range(rows * cols):
        ax = axes[d // cols][d % cols]
        if d >= dim:
            ax.axis("off")
            continue
        ax.plot(x, history[:, d], color=_SERIES_BLUE, linewidth=1.6)
        ax.set_title(f"{ylabel}[{d}]", fontsize=8)
        ax.grid(True, color=_GRID_GRAY, linewidth=0.6)
        ax.tick_params(labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.supxlabel("inference step", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


class PolicyLogger(_base_policy.BasePolicy):
    """Wraps a policy and logs each inference's obs and predicted action chunk
    in a human-readable way: images as PNG, everything else as JSON."""

    def __init__(self, policy: _base_policy.BasePolicy, log_dir: str):
        self._policy = policy
        # Put each server run in its own timestamped subdirectory so runs don't
        # clobber each other's step_* dirs.
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_dir = pathlib.Path(log_dir) / run_id
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        # Rolling history of the executed action (first action of each chunk) and
        # the observed state, used to (re)draw the cumulative trajectory chart.
        self._action_hist: list[np.ndarray] = []
        self._state_hist: list[np.ndarray] = []
        logging.info("Logging inference obs/chunks to: %s", self._log_dir)

    @staticmethod
    def _find_state(flat: dict) -> np.ndarray | None:
        """Pull the 1-D proprioceptive state vector out of the flattened obs."""
        for key, value in flat.items():
            if not key.startswith("obs/") or "state" not in key.lower():
                continue
            arr = np.asarray(value)
            if arr.dtype.kind in "fiu" and arr.ndim == 1:
                return arr.astype(np.float32)
        return None

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        # Logging must NEVER break inference: any failure here would otherwise
        # propagate to the websocket handler, which closes the connection and
        # leaves the robot client hanging. Swallow everything and warn instead.
        try:
            self._log_step(obs, results)
        except Exception:
            logging.exception("Inference logging failed at step %d (ignored)", self._step)
        return results

    def _log_step(self, obs: dict, results: dict) -> None:
        step_dir = self._log_dir / f"step_{self._step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        self._step += 1

        flat = flax.traverse_util.flatten_dict(
            {"obs": obs, "actions": results["actions"]}, sep="/")

        summary: dict = {}
        for key, value in flat.items():
            if isinstance(value, (bytes, bytearray)):
                # Images usually arrive as raw encoded (JPEG/PNG) bytes — decode and
                # save them as PNG instead of recording just the byte count.
                if key.startswith("obs/"):
                    try:
                        img = np.asarray(Image.open(io.BytesIO(bytes(value))).convert("RGB"))
                        if _is_image(img):
                            name = key.replace("/", "_") + ".png"
                            _save_image(img, step_dir / name)
                            summary[key] = {"image": name, "shape": list(img.shape),
                                            "bytes": len(value)}
                            continue
                    except Exception:
                        pass
                # Not an image: prompt / text as utf-8, else just record the size.
                try:
                    summary[key] = value.decode("utf-8")
                except UnicodeDecodeError:
                    summary[key] = {"bytes": len(value)}
                continue
            arr = np.asarray(value)
            orig_dtype = arr.dtype
            # Clients often send images as encoded JPEG/PNG bytes, which arrive as a
            # bytes/str/object-dtype array (kind S/U/O). Decode them back to uint8 so
            # _is_image fires and they get saved as PNG instead of dumped as raw bytes.
            if key.startswith("obs/") and arr.dtype.kind in "SUO":
                try:
                    decoded = np.squeeze(image_tools._maybe_decode_encoded(arr))
                    if decoded.dtype.kind in "fiu" and _is_image(decoded):
                        arr = decoded
                except Exception:
                    pass
            if key.startswith("obs/") and arr.dtype.kind in "fiu" and _is_image(arr):
                name = key.replace("/", "_") + ".png"
                _save_image(arr, step_dir / name)
                summary[key] = {"image": name, "shape": list(arr.shape), "dtype": str(orig_dtype)}
            elif arr.dtype.kind in "SUO":  # bytes / str / object array — not JSON-safe
                summary[key] = arr.astype(str).tolist()
            else:
                summary[key] = arr.tolist()

        with open(step_dir / "data.json", "w") as f:
            json.dump(summary, f, indent=2, default=lambda o: repr(o))

        self._render_charts(flat, results, step_dir)

    def _render_charts(self, flat: dict, results: dict, step_dir: pathlib.Path) -> None:
        """Draw the intuitive views: this chunk as a heatmap, plus the cumulative
        state / executed-action trajectory (overwritten at the run root)."""
        actions = np.asarray(results["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]

        # Prompt (if the client sent one) makes a useful chart title.
        prompt = ""
        for key, value in flat.items():
            if key.endswith("prompt") and isinstance(value, (bytes, bytearray, str)):
                prompt = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
                break
        title = f"step {self._step - 1}  ({actions.shape[0]}×{actions.shape[1]})"
        if prompt:
            title += f"  —  {prompt[:60]}"

        # _render_chunk(actions, step_dir / "chunk.png", title)

        # Executed action = first action of the chunk; state from the obs.
        self._action_hist.append(actions[0])
        state = self._find_state(flat)
        if state is not None:
            self._state_hist.append(state)

        _render_trajectory(
            np.stack(self._action_hist), self._log_dir / "trajectory_action.png",
            "action", "Executed action (first of each chunk) over inference steps",
        )
        if self._state_hist:
            _render_trajectory(
                np.stack(self._state_hist), self._log_dir / "trajectory_state.png",
                "state", "Observed state over inference steps",
            )

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            data_config = train_config.data.create(
                train_config.assets_dirs, train_config.model)

            # Serving has no ground-truth action; drop the "actions<-action" mapping
            # so RepackTransform doesn't look for a key the client never sends.
            repack = data_config.repack_transforms
            infer_repack = transforms.Group(
                inputs=[
                    transforms.RepackTransform({
                        k: v for k, v in t.structure.items() if k != "actions"
                    }) if isinstance(t, transforms.RepackTransform) else t
                    for t in repack.inputs
                ],
                outputs=repack.outputs,
            )
            return _policy_config.create_trained_policy(
                train_config, args.policy.dir,
                repack_transforms=infer_repack,
                default_prompt=args.default_prompt,
                sample_kwargs={"num_steps": args.num_steps},
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Log each inference's obs and predicted action chunk.
    policy = PolicyLogger(policy, args.log_dir)

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
