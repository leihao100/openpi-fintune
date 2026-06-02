"""Unitree G1 Dex1 input/output transforms for openpi.

Dataset: LeRobot v2.1, Unitree_G1_Dex1_Sim
    observation.state                       float32 (16,)  L_arm(7)+R_arm(7)+L_grip(1)+R_grip(1)
    action                                  float32 (16,)
    observation.images.cam_left_high        video 480x640 RGB
    observation.images.cam_left_wrist       video 480x640 RGB
    observation.images.cam_right_wrist      video 480x640 RGB
"""
import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 HWC. Handles both float32 CHW [0,1] (LeRobot
    video decode) and uint8 HWC (live inference)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# RepackTransform structure: {output_key: lerobot_dataset_key}
# NOTE: source key is "action" (singular), as stored by LeRobot.
REPACK_STRUCTURE = {
    "observation/image":             "observation.images.cam_left_high",
    "observation/left_wrist_image":  "observation.images.cam_left_wrist",
    "observation/right_wrist_image": "observation.images.cam_right_wrist",
    "observation/state":             "observation.state",
    "actions":                       "action",
    "prompt":                        "prompt",
}

# State/action layout in the LeRobot dataset:
#   [0:7]   left arm joints  (ShoulderPitch, ShoulderRoll, ShoulderYaw,
#                             Elbow, WristRoll, WristPitch, WristYaw)
#   [7:14]  right arm joints (same order)
#   [14]    left gripper
#   [15]    right gripper
STATE_DIM = 16
ACTION_DIM = 16


@dataclasses.dataclass(frozen=True)
class UnitreeG1Inputs(transforms.DataTransformFn):
    """Maps Unitree G1 LeRobot dataset fields to openpi model inputs."""

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
class UnitreeG1Outputs(transforms.DataTransformFn):
    """Extracts Unitree G1 actions (16 dims) from model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}