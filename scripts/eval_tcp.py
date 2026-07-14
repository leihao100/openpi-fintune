"""TCP-space eval: run a trained policy over one dataset episode and compare its
predicted action chunk against the demo (ground-truth) actions **in end-effector
(TCP) space** instead of raw joint space.

The Unitree G1 policy predicts a 16-D action:
    [0:7]  left arm joints   (ShoulderPitch, ShoulderRoll, ShoulderYaw,
                              Elbow, WristRoll, WristPitch, WristYaw)
    [7:14] right arm joints  (same order)
    [14]   left gripper
    [15]   right gripper

This script runs forward kinematics (the same reduced pinocchio model that
`teleop/robot_control/robot_arm_ik.py::G1_29_ArmIK` uses for IK) on the 14 arm
joints to recover the left/right wrist TCP pose, then shows the model's effect as
a 6-D (xyz + rpy) or 7-D (xyz + quat) TCP per arm.

Two stages, run in two different environments (kept in one file):
  * `infer` — needs the openpi `uv` env (JAX/torch + the policy). Runs the policy
    and dumps demo/pred action chunks (T, H, 16) to an intermediate .npz.
  * `fk`    — needs pinocchio (isolated `~/pin_fk_env`). Loads the .npz, does FK,
    and writes the TCP comparison plots.

`--stage auto` (default) runs `infer` in the current interpreter, then shells out
to the pinocchio interpreter (`--fk_python`) to run `fk`.

Example (one command, from /home/ur3-exp/pi/openpi):
    uv run scripts/eval_tcp.py \
        --config pi05_unitree_g1 \
        --checkpoint checkpoints/pi05_unitree_g1/exp/16000 \
        --episode 0 --tcp 6d --out_dir eval_tcp_out
"""

import dataclasses
import logging
import pathlib
import subprocess
import sys

import numpy as np
import tyro


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
    # Directory to write the comparison plots (and the intermediate .npz) into.
    out_dir: str = "eval_tcp_out"
    # Evaluate every `stride`-th frame (1 = every frame).
    stride: int = 1
    # Optional cap on the number of frames evaluated.
    max_frames: int | None = None
    # Draw a predicted action chunk every `chunk_every` frames in the raw-output plot.
    chunk_every: int = 5

    # TCP representation: "6d" = xyz + rpy(3), "7d" = xyz + quat(4).
    tcp: str = "6d"
    # Rotation encoding for the 6d TCP: "rpy" (roll/pitch/yaw) or "axisangle" (log3).
    rot: str = "rpy"

    # --- Forward-kinematics model (mirrors G1_29_ArmIK) ---
    urdf: str = "/home/ur3-exp/unitree/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    model_dir: str = "/home/ur3-exp/unitree/xr_teleoperate/assets/g1/"
    # x-offset (m) of the L_ee/R_ee frame from the wrist_yaw joint (matches the IK).
    ee_offset: float = 0.05

    # --- Stage plumbing ---
    # "auto" | "infer" | "fk". auto: run infer here, then shell out to fk_python for fk.
    stage: str = "auto"
    # Interpreter that has pinocchio installed (for the fk stage).
    fk_python: str = "/home/ur3-exp/pin_fk_env/bin/python"
    # Intermediate .npz path. Default: <out_dir>/episode_<ep>_actions.npz.
    npz: str | None = None


# Joints locked when building the reduced arm-only model — identical to
# G1_29_ArmIK.mixed_jointsToLockIDs so FK matches the IK solver exactly.
_LOCK_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

N_ARM = 14  # 7 left + 7 right


# --------------------------------------------------------------------------- #
# Stage 1: infer (openpi env) — dump demo/pred action chunks to .npz
# --------------------------------------------------------------------------- #
def run_infer(args: Args, npz_path: pathlib.Path) -> None:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import torch

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    import openpi.transforms as _transforms

    def to_obs(frame: dict) -> dict:
        return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in frame.items()}

    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    action_horizon = train_config.model.action_horizon
    action_key = data_config.action_sequence_keys[0]

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
        out = policy.infer(to_obs(frame))
        demo.append(np.asarray(frame[action_key]))
        pred.append(np.asarray(out["actions"]))

    demo = np.stack(demo)  # (T, H, D)
    pred = np.stack(pred)  # (T, H, D)
    d = min(demo.shape[-1], pred.shape[-1])
    demo, pred = demo[..., :d], pred[..., :d]

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        demo=demo.astype(np.float32),
        pred=pred.astype(np.float32),
        indices=np.asarray(indices, dtype=np.int64),
        start=np.int64(start),
        action_horizon=np.int64(action_horizon),
    )
    logging.info("Wrote joint-space action chunks to %s  (demo %s, pred %s)",
                 npz_path, demo.shape, pred.shape)


# --------------------------------------------------------------------------- #
# Stage 2: fk (pinocchio env) — FK -> TCP, then plot
# --------------------------------------------------------------------------- #
def _build_fk_model(args: Args):
    import pinocchio as pin

    robot = pin.RobotWrapper.BuildFromURDF(args.urdf, args.model_dir)
    reduced = robot.buildReducedRobot(
        list_of_joints_to_lock=_LOCK_JOINTS,
        reference_configuration=np.zeros(robot.model.nq),
    )
    off = pin.SE3(np.eye(3), np.array([args.ee_offset, 0.0, 0.0]))
    reduced.model.addFrame(
        pin.Frame("L_ee", reduced.model.getJointId("left_wrist_yaw_joint"), off, pin.FrameType.OP_FRAME))
    reduced.model.addFrame(
        pin.Frame("R_ee", reduced.model.getJointId("right_wrist_yaw_joint"), off, pin.FrameType.OP_FRAME))
    if reduced.model.nq != N_ARM:
        raise RuntimeError(f"reduced model nq={reduced.model.nq}, expected {N_ARM}")
    return reduced.model


def _se3_to_tcp(M, tcp: str, rot: str):
    """SE3 -> TCP vector. 6d: xyz+rpy|axisangle(3); 7d: xyz+quat(4, w>=0)."""
    import pinocchio as pin

    t = M.translation
    if tcp == "7d":
        q = pin.Quaternion(M.rotation)
        v = np.array([q.x, q.y, q.z, q.w])
        if v[3] < 0:  # canonicalize sign so plots stay continuous
            v = -v
        return np.concatenate([t, v])
    if rot == "axisangle":
        r = pin.log3(M.rotation)
    else:
        r = pin.rpy.matrixToRpy(M.rotation)
    return np.concatenate([t, r])


def _actions_to_tcp(actions: np.ndarray, model, args: Args) -> np.ndarray:
    """(N, 16) joint actions -> (N, tcp_dim) TCP. tcp_dim = 14 (6d) or 16 (7d):
    [L_tcp, R_tcp, L_grip, R_grip]."""
    import pinocchio as pin

    data = model.createData()
    lid, rid = model.getFrameId("L_ee"), model.getFrameId("R_ee")
    out = []
    for a in actions:
        q = np.asarray(a[:N_ARM], dtype=np.float64)
        pin.framesForwardKinematics(model, data, q)
        l = _se3_to_tcp(data.oMf[lid], args.tcp, args.rot)
        r = _se3_to_tcp(data.oMf[rid], args.tcp, args.rot)
        out.append(np.concatenate([l, r, a[14:16]]))
    return np.stack(out)


def _tcp_labels(tcp: str, rot: str) -> list[str]:
    if tcp == "7d":
        per = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    else:
        per = ["x", "y", "z"] + (["rx", "ry", "rz"] if rot == "axisangle" else ["roll", "pitch", "yaw"])
    return [f"L_{p}" for p in per] + [f"R_{p}" for p in per] + ["L_grip", "R_grip"]


def _geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Angle (deg) between two [x,y,z,w] quaternion arrays, shape (...,4)."""
    dot = np.abs(np.sum(qa * qb, axis=-1)).clip(-1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def run_fk(args: Args, npz_path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path)
    demo_j, pred_j = z["demo"], z["pred"]           # (T, H, 16)
    indices, start = z["indices"], int(z["start"])
    action_horizon = int(z["action_horizon"])
    T, H, _ = demo_j.shape

    model = _build_fk_model(args)
    # Convert both chunk tensors: flatten (T,H) -> FK -> reshape back.
    demo = _actions_to_tcp(demo_j.reshape(-1, demo_j.shape[-1]), model, args).reshape(T, H, -1)
    pred = _actions_to_tcp(pred_j.reshape(-1, pred_j.shape[-1]), model, args).reshape(T, H, -1)
    labels = _tcp_labels(args.tcp, args.rot)
    d = demo.shape[-1]

    abs_err = np.abs(pred - demo)
    logging.info("TCP MAE=%.5f  MSE=%.5f", abs_err.mean(), np.square(pred - demo).mean())

    # Physically meaningful errors on the executed action (chunk step 0):
    #   position: euclidean over xyz (m -> mm); orientation: geodesic angle (deg).
    def pos_err_mm(side_off):
        return np.linalg.norm(pred[:, 0, side_off:side_off + 3] - demo[:, 0, side_off:side_off + 3], axis=-1) * 1000.0

    per = 7 if args.tcp == "7d" else 6
    l_pos = pos_err_mm(0)
    r_pos = pos_err_mm(per)
    if args.tcp == "7d":
        l_rot = _geodesic_deg(pred[:, 0, 3:7], demo[:, 0, 3:7])
        r_rot = _geodesic_deg(pred[:, 0, per + 3:per + 7], demo[:, 0, per + 3:per + 7])
        logging.info("L pos %.1f mm | rot %.2f deg   R pos %.1f mm | rot %.2f deg (mean)",
                     l_pos.mean(), l_rot.mean(), r_pos.mean(), r_rot.mean())
    else:
        logging.info("L pos %.1f mm   R pos %.1f mm (mean)", l_pos.mean(), r_pos.mean())

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"episode_{args.episode}_tcp{args.tcp}"
    t = np.asarray(indices) - start
    ncols = min(4, d)
    nrows = -(-d // ncols)

    # 1) Per-dim executed TCP (chunk step 0): demo vs predicted.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for j in range(d):
        ax = axes[j // ncols][j % ncols]
        ax.plot(t, demo[:, 0, j], label="demo", color="tab:blue")
        ax.plot(t, pred[:, 0, j], label="pred", color="tab:orange", linestyle="--")
        ax.set_title(f"{labels[j]}  (MAE {abs_err[:, 0, j].mean():.4f})")
        ax.set_xlabel("frame")
    axes[0][0].legend()
    for k in range(d, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{args.config}  ep {args.episode}  —  executed TCP ({args.tcp}, chunk step 0)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_pose.png", dpi=120)
    plt.close(fig)

    # 2) Physical per-frame error: position (mm) and, for 7d, orientation (deg).
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, l_pos, color="tab:blue", label=f"L pos (mean {l_pos.mean():.1f} mm)")
    ax.plot(t, r_pos, color="tab:green", label=f"R pos (mean {r_pos.mean():.1f} mm)")
    ax.set_ylabel("position error (mm)")
    ax.set_xlabel("frame")
    if args.tcp == "7d":
        ax2 = ax.twinx()
        ax2.plot(t, l_rot, color="tab:blue", linestyle=":", alpha=0.7,
                 label=f"L rot (mean {l_rot.mean():.1f} deg)")
        ax2.plot(t, r_rot, color="tab:green", linestyle=":", alpha=0.7,
                 label=f"R rot (mean {r_rot.mean():.1f} deg)")
        ax2.set_ylabel("orientation error (deg)")
        ax2.legend(loc="upper right")
    ax.legend(loc="upper left")
    ax.set_title(f"per-frame TCP error  (ep {args.episode})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_error.png", dpi=120)
    plt.close(fig)

    # 3) Raw output: overlay every predicted TCP chunk on the demo trajectory.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for j in range(d):
        ax = axes[j // ncols][j % ncols]
        ax.plot(t, demo[:, 0, j], color="tab:blue", linewidth=1.5, label="demo", zorder=3)
        for i in range(0, len(t), args.chunk_every):
            ax.plot(t[i] + np.arange(action_horizon), pred[i, :, j],
                    color="tab:orange", alpha=0.35, linewidth=0.8, zorder=2)
        ax.set_title(labels[j])
        ax.set_xlabel("frame")
    axes[0][0].legend()
    for k in range(d, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{args.config}  ep {args.episode}  —  raw predicted TCP chunks vs demo")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_chunks.png", dpi=120)
    plt.close(fig)

    logging.info("Saved TCP plots to %s", out_dir)


def main(args: Args) -> None:
    if args.tcp not in ("6d", "7d"):
        raise ValueError("--tcp must be '6d' or '7d'")
    if args.rot not in ("rpy", "axisangle"):
        raise ValueError("--rot must be 'rpy' or 'axisangle'")

    npz_path = pathlib.Path(args.npz) if args.npz else \
        pathlib.Path(args.out_dir) / f"episode_{args.episode}_actions.npz"

    if args.stage in ("auto", "infer"):
        run_infer(args, npz_path)

    if args.stage == "fk":
        run_fk(args, npz_path)
    elif args.stage == "auto":
        # Shell out to the pinocchio interpreter for the FK+plot stage.
        cmd = [args.fk_python, __file__, "--stage", "fk",
               "--config", args.config, "--checkpoint", args.checkpoint,
               "--episode", str(args.episode), "--out_dir", args.out_dir,
               "--tcp", args.tcp, "--rot", args.rot,
               "--urdf", args.urdf, "--model_dir", args.model_dir,
               "--ee_offset", str(args.ee_offset),
               "--chunk_every", str(args.chunk_every), "--npz", str(npz_path)]
        logging.info("Running FK stage: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
