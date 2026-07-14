"""Unitree G1 Dex1 EEF-space transforms using a 6D rotation action space.

Motivation
----------
The quaternion EEF policy (`unitree_eef_policy.py`) predicts orientation as a raw
(qx, qy, qz, qw). Quaternions are a poor *regression target* for a VLA:
  * double cover — q and -q are the same rotation, so nearby visual states can
    map to opposite-sign targets across episodes (episode-local sign continuity
    does NOT give a global convention), forcing the model to fit a bimodal target;
  * predictions are not unit-norm, and the map is most sensitive exactly near
    the identity rotation where most of the data lives.
Empirically the left-arm quaternion dims were the worst-tracked channels.

The fix is the continuous 6D rotation representation (Zhou et al., CVPR 2019):
the first two columns of the rotation matrix. It is a *function of the rotation
matrix only*, so q and -q collapse to the SAME 6D vector — the double-cover
ambiguity disappears for free, and there is no unit-norm constraint to violate.

Representation contract
-----------------------
On disk (and on the wire to/from the on-robot client) the state/action stay in
the 16-dim quaternion layout the datasets and `main_eef.py` already use:

    quat16 = [ L xyz(3) | L quat xyzw(4) | R xyz(3) | R quat xyzw(4) | Lgrip | Rgrip ]

The model, however, sees/produces a 20-dim 6D layout:

    sixd20 = [ L xyz(3) | L rot6d(6) | R xyz(3) | R rot6d(6) | Lgrip | Rgrip ]

`UnitreeG1EEF6DInputs`  converts quat16 -> sixd20 (state + actions) before the
model, so norm_stats are computed on the 6D representation.
`UnitreeG1EEF6DOutputs` converts the model's sixd20 action chunk back to quat16,
so the on-robot client and its IK (`eef_kinematics.py`) need NO changes.
"""
import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """uint8 HWC. float32 [0,1] CHW (video decode) or uint8 HWC (live)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# ---- quaternion <-> 6D rotation (batched over any leading axes) ----

def _quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternion -> (..., 6) = first two rotation-matrix columns.

    q and -q give the same rotation matrix, hence the same 6D vector: this is
    exactly what removes the double-cover ambiguity across episodes."""
    quat = np.asarray(quat, dtype=np.float64)
    flat = quat.reshape(-1, 4)
    norms = np.linalg.norm(flat, axis=-1, keepdims=True)
    flat = flat / np.where(norms < 1e-8, 1.0, norms)
    mats = Rotation.from_quat(flat).as_matrix()          # (N, 3, 3), scipy = xyzw
    six = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=-1)  # [col0 | col1]
    return six.reshape(quat.shape[:-1] + (6,)).astype(np.float32)


def _rot6d_to_quat(six: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 4) xyzw, via Gram-Schmidt (Zhou et al. 2019)."""
    six = np.asarray(six, dtype=np.float64)
    flat = six.reshape(-1, 6)
    a1, a2 = flat[:, :3], flat[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    mats = np.stack([b1, b2, b3], axis=-1)               # columns -> (N, 3, 3)
    quat = Rotation.from_matrix(mats).as_quat()          # (N, 4) xyzw
    return quat.reshape(six.shape[:-1] + (4,)).astype(np.float32)


# quat16 field slices
_Q_L_POS, _Q_L_ROT = slice(0, 3), slice(3, 7)
_Q_R_POS, _Q_R_ROT = slice(7, 10), slice(10, 14)
_Q_LGRIP, _Q_RGRIP = 14, 15
# sixd20 field slices
_S_L_POS, _S_L_ROT = slice(0, 3), slice(3, 9)
_S_R_POS, _S_R_ROT = slice(9, 12), slice(12, 18)
_S_LGRIP, _S_RGRIP = 18, 19

QUAT_DIM = 16
SIXD_DIM = 20


def quat16_to_sixd20(x: np.ndarray) -> np.ndarray:
    """(..., 16) quaternion layout -> (..., 20) 6D layout."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty(x.shape[:-1] + (SIXD_DIM,), dtype=np.float32)
    out[..., _S_L_POS] = x[..., _Q_L_POS]
    out[..., _S_L_ROT] = _quat_to_rot6d(x[..., _Q_L_ROT])
    out[..., _S_R_POS] = x[..., _Q_R_POS]
    out[..., _S_R_ROT] = _quat_to_rot6d(x[..., _Q_R_ROT])
    out[..., _S_LGRIP] = x[..., _Q_LGRIP]
    out[..., _S_RGRIP] = x[..., _Q_RGRIP]
    return out


def sixd20_to_quat16(x: np.ndarray) -> np.ndarray:
    """(..., 20) 6D layout -> (..., 16) quaternion layout."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty(x.shape[:-1] + (QUAT_DIM,), dtype=np.float32)
    out[..., _Q_L_POS] = x[..., _S_L_POS]
    out[..., _Q_L_ROT] = _rot6d_to_quat(x[..., _S_L_ROT])
    out[..., _Q_R_POS] = x[..., _S_R_POS]
    out[..., _Q_R_ROT] = _rot6d_to_quat(x[..., _S_R_ROT])
    out[..., _Q_LGRIP] = x[..., _S_LGRIP]
    out[..., _Q_RGRIP] = x[..., _S_RGRIP]
    return out


@dataclasses.dataclass(frozen=True)
class UnitreeG1EEF6DInputs(transforms.DataTransformFn):
    """quat16 dataset/client fields -> 6D (20-dim) model inputs."""

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = quat16_to_sixd20(np.asarray(data["observation/state"], dtype=np.float32))

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb":        _parse_image(data["observation/image"]),
                "left_wrist_0_rgb":  _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            "image_mask": {
                "base_0_rgb":        np.True_,
                "left_wrist_0_rgb":  np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = quat16_to_sixd20(np.asarray(data["actions"], dtype=np.float32))
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UnitreeG1EEF6DOutputs(transforms.DataTransformFn):
    """Model 6D (20-dim) action chunk -> quat16 for the on-robot client.

    Slices the first 20 dims (the model pads action_dim to 32) and maps each
    arm's 6D rotation back to a unit quaternion, so the returned action matches
    the [H, 16] quaternion contract main_eef.py / eef_kinematics.py expect."""

    def __call__(self, data: dict) -> dict:
        return {"actions": sixd20_to_quat16(np.asarray(data["actions"][:, :SIXD_DIM]))}
