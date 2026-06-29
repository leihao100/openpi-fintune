"""Data replay: load one dataset episode and visualize it as charts —
per-dim state and action trajectories plus a strip of camera frames.

Unlike eval.py this runs no policy; it only inspects the recorded data.

Example:
    python scripts/data_replay.py --config pi05_unitree_g1 \
        --episode 0 --out_dir replay_out
"""

import dataclasses
import logging
import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # Train config name (e.g. "pi05_unitree_g1"); used to resolve the dataset path.
    config: str
    # Dataset directory to replay. Overrides the path derived from the config.
    data_path: str | None = None
    # Episode index within the dataset to replay.
    episode: int = 0
    # Directory to write the plots into.
    out_dir: str = "replay_out"
    # Replay every `stride`-th frame (1 = every frame).
    stride: int = 1
    # Optional cap on the number of frames replayed.
    max_frames: int | None = None
    # Number of camera frames to lay out in the image strip.
    num_images: int = 6
    # Only plot actions; skip the state and camera-frame plots.
    action_only: bool = False


def _state_key(meta: lerobot_dataset.LeRobotDatasetMetadata) -> str | None:
    for k in ("observation.state", "state"):
        if k in meta.features:
            return k
    return None


def _image_keys(meta: lerobot_dataset.LeRobotDatasetMetadata) -> list[str]:
    keys = []
    for k, feat in meta.features.items():
        if feat.get("dtype") in ("image", "video") or "image" in k:
            keys.append(k)
    return keys


def _as_np(v) -> np.ndarray:
    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _plot_series(t: np.ndarray, data: np.ndarray, title: str, path: pathlib.Path) -> None:
    """data: (T, D) -> one subplot per dim."""
    d = data.shape[-1]
    ncols = min(4, d)
    nrows = -(-d // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for j in range(d):
        ax = axes[j // ncols][j % ncols]
        ax.plot(t, data[:, j], color="tab:blue")
        ax.set_title(f"dim {j}  [{data[:, j].min():.3f}, {data[:, j].max():.3f}]")
        ax.set_xlabel("frame")
    for k in range(d, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    action_key = data_config.action_sequence_keys[0]

    if args.data_path is not None:
        root = pathlib.Path(args.data_path)
    elif data_config.local_root:
        root = data_config.local_root / data_config.repo_id
    else:
        root = None

    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=root)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, root=root)

    state_key = None if args.action_only else _state_key(meta)
    image_keys = [] if args.action_only else _image_keys(meta)
    logging.info("state=%s  action=%s  images=%s", state_key, action_key, image_keys)

    start = int(dataset.episode_data_index["from"][args.episode])
    end = int(dataset.episode_data_index["to"][args.episode])
    indices = list(range(start, end, args.stride))
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    logging.info("Episode %d: frames %d-%d (%d replayed)", args.episode, start, end, len(indices))

    states, actions = [], []
    for idx in indices:
        frame = dataset[idx]
        if state_key is not None:
            states.append(_as_np(frame[state_key]).reshape(-1))
        actions.append(_as_np(frame[action_key]).reshape(-1))

    actions = np.stack(actions)  # (T, D)
    t = np.asarray(indices) - start

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if states:
        states = np.stack(states)  # (T, D)
        _plot_series(t, states, f"{args.config}  episode {args.episode}  —  state",
                     out_dir / f"episode_{args.episode}_state.png")
    _plot_series(t, actions, f"{args.config}  episode {args.episode}  —  action",
                 out_dir / f"episode_{args.episode}_action.png")

    # Camera frames evenly sampled across the episode.
    for cam in image_keys:
        sample = np.linspace(0, len(indices) - 1, min(args.num_images, len(indices))).round().astype(int)
        fig, axes = plt.subplots(1, len(sample), figsize=(2.5 * len(sample), 2.8), squeeze=False)
        for col, s in enumerate(sample):
            img = _to_hwc_uint8(_as_np(dataset[indices[s]][cam]))
            ax = axes[0][col]
            ax.imshow(img)
            ax.set_title(f"frame {t[s]}")
            ax.axis("off")
        fig.suptitle(f"{args.config}  episode {args.episode}  —  {cam}")
        fig.tight_layout()
        safe = cam.replace(".", "_").replace("/", "_")
        fig.savefig(out_dir / f"episode_{args.episode}_{safe}.png", dpi=120)
        plt.close(fig)

    logging.info("Saved plots to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
