"""Unitree G1 Dex1 EEF-space input/output transforms for openpi.

Dataset: LeRobot v2.1, Unitree_G1_Dex1_Sim_EEF (converted from joint space by
~/unitree/data/joint_to_eef.py via forward kinematics, pelvis frame, waist locked at 0)
    observation.state                       float32 (16,)  L_eef(7)+R_eef(7)+L_grip(1)+R_grip(1)
    action                                  float32 (16,)
    observation.images.cam_left_high        video 480x640 RGB
    observation.images.cam_left_wrist       video 480x640 RGB
    observation.images.cam_right_wrist      video 480x640 RGB

EEF layout per arm: x, y, z, qx, qy, qz, qw (position in meters, unit quaternion,
sign-continuous along each episode). The EEF frame is wrist_yaw + 0.05 m x-offset,
matching xr_teleoperate's G1_29_ArmIK 'L_ee'/'R_ee' frames.
"""
import dataclasses
import einops
import numpy as np
import cv2

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 HWC. Handles:
      - JPEG-encoded bytes (client --send-jpeg): cv2.imdecode (-> BGR) then
        BGR->RGB, matching the client's native-BGR JPEG encode contract.
      - float32 CHW [0,1] (LeRobot video decode).
      - uint8 HWC (live inference, already-decoded RGB)."""
    # --send-jpeg path: client ships compressed JPEG bytes to cut upload ~12x.
    if isinstance(image, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(bytes(image), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = np.asarray(image)
    # JPEG bytes that arrived as a 1-D uint8 array (JPEG SOI marker 0xFFD8).
    if (image.ndim == 1 and image.dtype == np.uint8 and image.size > 2
            and image[0] == 0xFF and image[1] == 0xD8):
        bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# State/action layout in the EEF LeRobot dataset:
#   [0:3]   left EEF position  (x, y, z)      pelvis frame, meters
#   [3:7]   left EEF quaternion (qx, qy, qz, qw)
#   [7:10]  right EEF position
#   [10:14] right EEF quaternion
#   [14]    left gripper
#   [15]    right gripper
STATE_DIM = 16
ACTION_DIM = 16


@dataclasses.dataclass(frozen=True)
class UnitreeG1EEFInputs(transforms.DataTransformFn):
    """Maps Unitree G1 EEF-space LeRobot dataset fields to openpi model inputs."""

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)

        base_image        = _parse_image(data["observation/image"])
        left_wrist_image  = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb":        base_image,
                "left_wrist_0_rgb":  left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb":        np.True_,
                "left_wrist_0_rgb":  np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UnitreeG1EEFOutputs(transforms.DataTransformFn):
    """Extracts Unitree G1 EEF actions (16 dims) from model output.

    Note: predicted quaternions are not guaranteed to be unit-norm; consumers
    should normalize q[3:7] and q[10:14] before feeding them to an IK solver.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
