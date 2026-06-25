"""Frame-by-frame eval: run a trained policy over one dataset episode and
compare its predicted action chunk against the demo (ground-truth) actions.

Example:
    python scripts/eval.py --config pi05_unitree_g1 \
        --checkpoint checkpoints/pi05_unitree_g1/exp/16000 \
        --episode 0 --out_dir eval_out
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

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import openpi.transforms as _transforms


@dataclasses.dataclass
class Args:
    # Train config name (e.g. "pi05_unitree_g1").
    config: str
    # Checkpoint directory to load the trained policy from.
    checkpoint: str
    # Dataset directory to evaluate on. Overrides the path derived from the config.
    data_path: str | None = None
    # Episode index within the dataset to evaluate.
    episode: int = 0
    # Directory to write the comparison plots into.
    out_dir: str = "eval_out"
    # Evaluate every `stride`-th frame (1 = every frame).
    stride: int = 1
    # Optional cap on the number of frames evaluated.
    max_frames: int | None = None
    # Draw a predicted action chunk every `chunk_every` frames in the raw-output plot.
    chunk_every: int = 5


def _to_obs(frame: dict) -> dict:
    """Convert a raw LeRobot frame to numpy, leaving non-tensor values as-is."""
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in frame.items()}


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    action_horizon = train_config.model.action_horizon
    action_key = data_config.action_sequence_keys[0]

    # Raw dataset: each frame's action is a chunk of length `action_horizon`,
    # matching what the policy predicts.
    if args.data_path is not None:
        root = pathlib.Path(args.data_path)
    elif data_config.local_root:
        root = data_config.local_root / data_config.repo_id
    else:
        root = None
    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=root)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=root,
        delta_timestamps={action_key: [t / meta.fps for t in range(action_horizon)]},
    )
    prompt_fn = _transforms.PromptFromLeRobotTask(meta.tasks) if data_config.prompt_from_task else None

    # Feed raw frames straight into the policy by reusing the training repack.
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint, repack_transforms=data_config.repack_transforms
    )

    start = int(dataset.episode_data_index["from"][args.episode])
    end = int(dataset.episode_data_index["to"][args.episode])
    indices = list(range(start, end, args.stride))
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    logging.info("Episode %d: frames %d-%d (%d evaluated)", args.episode, start, end, len(indices))

    demo, pred = [], []
    for idx in indices:
        frame = dataset[idx]
        if prompt_fn is not None:
            frame = prompt_fn(frame)
        out = policy.infer(_to_obs(frame))
        demo.append(np.asarray(frame[action_key]))
        pred.append(np.asarray(out["actions"]))

    demo = np.stack(demo)  # (T, H, D)
    pred = np.stack(pred)  # (T, H, D)
    d = min(demo.shape[-1], pred.shape[-1])
    demo, pred = demo[..., :d], pred[..., :d]

    abs_err = np.abs(pred - demo)
    logging.info("MAE=%.5f  MSE=%.5f", abs_err.mean(), np.square(pred - demo).mean())

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-dim executed action (first step of each chunk): demo vs predicted.
    t = np.asarray(indices) - start
    ncols = min(4, d)
    nrows = -(-d // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for j in range(d):
        ax = axes[j // ncols][j % ncols]
        ax.plot(t, demo[:, 0, j], label="demo", color="tab:blue")
        ax.plot(t, pred[:, 0, j], label="pred", color="tab:orange", linestyle="--")
        ax.set_title(f"action dim {j}  (MAE {abs_err[:, 0, j].mean():.4f})")
        ax.set_xlabel("frame")
    axes[0][0].legend()
    for k in range(d, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{args.config}  episode {args.episode}  —  executed action (chunk step 0)")
    fig.tight_layout()
    fig.savefig(out_dir / f"episode_{args.episode}_actions.png", dpi=120)
    plt.close(fig)

    # 2) Per-frame error: mean abs error over the whole predicted chunk and dims.
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, abs_err.mean(axis=(1, 2)), color="tab:red")
    ax.set_title(f"per-frame chunk MAE  (overall {abs_err.mean():.4f})")
    ax.set_xlabel("frame")
    ax.set_ylabel("mean |pred - demo|")
    fig.tight_layout()
    fig.savefig(out_dir / f"episode_{args.episode}_error.png", dpi=120)
    plt.close(fig)

    # 3) Raw output: overlay every predicted action chunk on the demo trajectory.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for j in range(d):
        ax = axes[j // ncols][j % ncols]
        ax.plot(t, demo[:, 0, j], color="tab:blue", linewidth=1.5, label="demo", zorder=3)
        for i in range(0, len(t), args.chunk_every):
            ax.plot(t[i] + np.arange(action_horizon), pred[i, :, j],
                    color="tab:orange", alpha=0.35, linewidth=0.8, zorder=2)
        ax.set_title(f"action dim {j}")
        ax.set_xlabel("frame")
    axes[0][0].legend()
    for k in range(d, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{args.config}  episode {args.episode}  —  raw predicted chunks vs demo")
    fig.tight_layout()
    fig.savefig(out_dir / f"episode_{args.episode}_chunks.png", dpi=120)
    plt.close(fig)

    logging.info("Saved plots to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
