import dataclasses
from typing import NotRequired, TypedDict

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Seeed B601 (single-arm follower) action/state layout (7 dims), degrees:
#   0  shoulder_pan
#   1  shoulder_lift
#   2  elbow_flex
#   3  wrist_flex
#   4  wrist_yaw
#   5  wrist_roll
#   6  gripper
# observation.state and action do NOT share a coordinate convention: four dims are sign-flipped and
# the gripper uses a different unit (state spans [-270, 0], action spans [0, 57]). See the internal B601 notes, issue I1.
B601_ACTION_DIM = 7
B601_STATE_DIM = 7
B601_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
# Native camera resolution of the delivered dataset; ResizeImages scales to 224x224 downstream.
B601_IMAGE_HW = (480, 640)


class B601ModelInputs(TypedDict):
    """Key set that B601Inputs emits.

    Declared explicitly because transforms.DataTransformFn fixes the runtime type to a plain dict.
    """

    state: np.ndarray
    image: dict[str, np.ndarray]
    image_mask: dict[str, np.bool_]
    actions: NotRequired[np.ndarray]
    prompt: NotRequired[str]


def make_b601_example() -> dict:
    """Creates a random input example for the B601 policy (matches the inference key format)."""
    return {
        "observation/state": np.random.rand(B601_STATE_DIM),
        "observation/top": np.random.randint(256, size=(*B601_IMAGE_HW, 3), dtype=np.uint8),
        "observation/wrist": np.random.randint(256, size=(*B601_IMAGE_HW, 3), dtype=np.uint8),
        "prompt": "Pick up the red cube and place it completely inside the tray.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B601Inputs(transforms.DataTransformFn):
    """Converts B601 observations into the format expected by the model.

    The B601 rig has only two cameras. The unused right wrist view is zero-filled and masked out, so
    its tokens are dropped from the prefix attention (see pi0.py embed_prefix).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/top"])
        wrist_image = _parse_image(data["observation/wrist"])

        # pi0-FAST has no image mask, so the padding view must be marked present there.
        mask_padding = self.model_type != _model.ModelType.PI0_FAST

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class B601Outputs(transforms.DataTransformFn):
    """Converts model actions back to the B601 7-dim joint command vector."""

    def __call__(self, data: dict) -> dict:
        # Strip the zero padding that PadStatesAndActions added to reach the model action dim.
        return {"actions": np.asarray(data["actions"][..., :B601_ACTION_DIM])}
