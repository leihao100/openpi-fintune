import dataclasses
import enum
import json
import logging
import pathlib
import socket

import flax.traverse_util
import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
import tyro
from typing_extensions import override

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


class PolicyLogger(_base_policy.BasePolicy):
    """Wraps a policy and logs each inference's obs and predicted action chunk
    in a human-readable way: images as PNG, everything else as JSON."""

    def __init__(self, policy: _base_policy.BasePolicy, log_dir: str):
        self._policy = policy
        self._log_dir = pathlib.Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        logging.info("Logging inference obs/chunks to: %s", self._log_dir)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        step_dir = self._log_dir / f"step_{self._step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        self._step += 1

        flat = flax.traverse_util.flatten_dict(
            {"obs": obs, "actions": results["actions"]}, sep="/")

        summary: dict = {}
        for key, value in flat.items():
            arr = np.asarray(value)
            if arr.dtype.kind in "fiu" and _is_image(arr):
                name = key.replace("/", "_") + ".png"
                _save_image(arr, step_dir / name)
                summary[key] = {"image": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
            else:
                summary[key] = arr.tolist()

        with open(step_dir / "data.json", "w") as f:
            json.dump(summary, f, indent=2)
        return results

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
