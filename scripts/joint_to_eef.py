#!/usr/bin/env python3
"""Convert a Unitree G1 LeRobot dataset from joint space to EEF (end-effector) space.

Input  state/action (16-dim): [left arm 7 joints, right arm 7 joints, left gripper, right gripper]
Output state/action (16-dim): [left EEF x,y,z,qx,qy,qz,qw, right EEF x,y,z,qx,qy,qz,qw, left gripper, right gripper]

The EEF pose is computed by forward kinematics with pinocchio, using the SAME model
definition as xr_teleoperate's G1_29_ArmIK: frames 'L_ee'/'R_ee' attached to
left/right_wrist_yaw_joint with a +0.05 m x-offset. Poses are expressed in the robot
pelvis frame, with waist/legs/fingers locked at 0 (matching teleop assumptions).

Usage:
    python joint_to_eef.py <input_dataset_dir> <output_dataset_dir> [--urdf PATH] [--assets DIR]

Example:
    ~/miniconda3/envs/lerobot/bin/python joint_to_eef.py stack-cube stack-cube-eef
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pinocchio as pin

DEFAULT_URDF = "/home/ur3-exp/unitree/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
DEFAULT_ASSETS = "/home/ur3-exp/unitree/xr_teleoperate/assets/g1/"
EE_OFFSET = 0.05  # meters, along x of the wrist yaw frame (same as G1_29_ArmIK)

# Dataset column order (G1_29_JointArmIndex) -> URDF joint names
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

EEF_NAMES = [
    "kLeftEEF_x", "kLeftEEF_y", "kLeftEEF_z",
    "kLeftEEF_qx", "kLeftEEF_qy", "kLeftEEF_qz", "kLeftEEF_qw",
    "kRightEEF_x", "kRightEEF_y", "kRightEEF_z",
    "kRightEEF_qx", "kRightEEF_qy", "kRightEEF_qz", "kRightEEF_qw",
    "kLeftGripper", "kRightGripper",
]


class G1ArmFK:
    def __init__(self, urdf_path=DEFAULT_URDF, assets_dir=DEFAULT_ASSETS):
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, assets_dir)
        lock_names = [n for n in robot.model.names[1:] if n not in ARM_JOINT_NAMES]
        self.reduced = robot.buildReducedRobot(
            list_of_joints_to_lock=lock_names,
            reference_configuration=np.zeros(robot.model.nq),
        )
        model = self.reduced.model
        assert model.nq == 14, f"expected 14-dof reduced model, got nq={model.nq}"
        model.addFrame(pin.Frame(
            "L_ee", model.getJointId("left_wrist_yaw_joint"), 0,
            pin.SE3(np.eye(3), np.array([EE_OFFSET, 0, 0])), pin.FrameType.OP_FRAME))
        model.addFrame(pin.Frame(
            "R_ee", model.getJointId("right_wrist_yaw_joint"), 0,
            pin.SE3(np.eye(3), np.array([EE_OFFSET, 0, 0])), pin.FrameType.OP_FRAME))
        self.data = model.createData()
        self.model = model
        self.l_id = model.getFrameId("L_ee")
        self.r_id = model.getFrameId("R_ee")
        # dataset column i -> position in pinocchio q vector
        self.q_index = np.array(
            [model.joints[model.getJointId(n)].idx_q for n in ARM_JOINT_NAMES])

    def fk(self, joints14):
        """joints14 in dataset order -> (left_xyzquat7, right_xyzquat7), quat = (qx,qy,qz,qw)."""
        q = np.zeros(self.model.nq)
        q[self.q_index] = joints14
        pin.framesForwardKinematics(self.model, self.data, q)
        left = pin.SE3ToXYZQUAT(self.data.oMf[self.l_id])
        right = pin.SE3ToXYZQUAT(self.data.oMf[self.r_id])
        return left, right


def fix_quat_continuity(arr, quat_slices):
    """Flip quaternion signs in-place so consecutive frames stay on the same cover."""
    for sl in quat_slices:
        for t in range(1, len(arr)):
            if np.dot(arr[t - 1, sl], arr[t, sl]) < 0:
                arr[t, sl] = -arr[t, sl]


def convert_column(values, fk):
    """values: (T, 16) joint-space array -> (T, 16) eef-space array."""
    out = np.empty_like(values, dtype=np.float32)
    for t, row in enumerate(values):
        left, right = fk.fk(row[:14])
        out[t, :7] = left
        out[t, 7:14] = right
        out[t, 14:] = row[14:]
    fix_quat_continuity(out, [slice(3, 7), slice(10, 14)])
    return out


def episode_stats(arr):
    return {
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--assets", default=DEFAULT_ASSETS)
    args = parser.parse_args()

    in_dir, out_dir = os.path.abspath(args.input_dir), os.path.abspath(args.output_dir)
    if os.path.exists(out_dir):
        sys.exit(f"output dir already exists: {out_dir}")

    with open(os.path.join(in_dir, "meta", "info.json")) as f:
        info = json.load(f)
    for key in ("observation.state", "action"):
        assert info["features"][key]["shape"] == [16], f"{key} must be 16-dim"

    fk = G1ArmFK(args.urdf, args.assets)
    new_stats = {}  # episode_index -> {feature: stats}

    # convert parquet episodes
    data_root = os.path.join(in_dir, "data")
    for chunk in sorted(os.listdir(data_root)):
        out_chunk = os.path.join(out_dir, "data", chunk)
        os.makedirs(out_chunk, exist_ok=True)
        files = sorted(f for f in os.listdir(os.path.join(data_root, chunk)) if f.endswith(".parquet"))
        for i, fname in enumerate(files):
            df = pd.read_parquet(os.path.join(data_root, chunk, fname))
            ep_idx = int(df["episode_index"].iloc[0])
            ep_stats = {}
            for key in ("observation.state", "action"):
                converted = convert_column(np.stack(df[key].to_numpy()), fk)
                df[key] = list(converted)
                ep_stats[key] = episode_stats(converted)
            new_stats[ep_idx] = ep_stats
            df.to_parquet(os.path.join(out_chunk, fname))
            print(f"\r{chunk}: {i + 1}/{len(files)} episodes", end="", flush=True)
        print()

    # meta: update feature names, recompute state/action stats, copy the rest
    meta_out = os.path.join(out_dir, "meta")
    os.makedirs(meta_out, exist_ok=True)
    info["robot_type"] = info.get("robot_type", "") + "_EEF"
    for key in ("observation.state", "action"):
        info["features"][key]["names"] = [EEF_NAMES]
    with open(os.path.join(meta_out, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    stats_path = os.path.join(in_dir, "meta", "episodes_stats.jsonl")
    if os.path.exists(stats_path):
        with open(stats_path) as f_in, open(os.path.join(meta_out, "episodes_stats.jsonl"), "w") as f_out:
            for line in f_in:
                entry = json.loads(line)
                entry["stats"].update(new_stats[entry["episode_index"]])
                f_out.write(json.dumps(entry) + "\n")

    for fname in os.listdir(os.path.join(in_dir, "meta")):
        if fname not in ("info.json", "episodes_stats.jsonl"):
            shutil.copy2(os.path.join(in_dir, "meta", fname), os.path.join(meta_out, fname))

    # videos: hard-link (same filesystem) to avoid duplicating storage
    videos_root = os.path.join(in_dir, "videos")
    if os.path.isdir(videos_root):
        for root, _, files in os.walk(videos_root):
            for fname in files:
                src = os.path.join(root, fname)
                link_or_copy(src, os.path.join(out_dir, "videos", os.path.relpath(src, videos_root)))

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
