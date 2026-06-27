# openpi 

This repository is forked from [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi). It adds data adapters, training configs, local dataset loading, open-loop evaluation, and inference-serving enhancements for two robot platforms: **Unitree G1 Dex1 (16-DoF bimanual)** and **UR3 (7-DoF single arm)**.



---

## 1. Run Commands

Examples use the Unitree G1 config `pi05_unitree_g1` (for UR3, swap the config name to `pi05_ur3_5task`, etc.; see all configs in [src/openpi/training/config.py](src/openpi/training/config.py)).

### 1.1 Split an eval set (optional)

Randomly sample episodes from a LeRobot v2.1 dataset into a separate eval dataset:

```bash
# Copy only (training set stays unchanged)
uv run scripts/split_dataset.py \
    --src /home/bioprocessing-lab/yuhao/data/put_cup_n_broccoli \
    --eval_dir /home/bioprocessing-lab/yuhao/data/put_cup_n_broccoli_eval \
    --num 5 --seed 0

# Add --move to also remove the selected episodes from the source (true held-out split)
```

### 1.2 Compute normalization statistics

Required before training (writes to `assets/<config>/<repo_id>/norm_stats.json`):

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_unitree_g1
```


### 1.3 Train

```bash
# Standard training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_unitree_g1 --exp-name=v1 --overwrite

# Training with periodic open-loop eval (uses the eval_* fields in the config;
# logs MAE/MSE and demo-vs-pred plots to wandb during training)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_eval.py pi05_unitree_g1 --exp-name=v1 --overwrite
```
Remember to use local_dir parameter in config.py, otherwise it won't save assets to yout ckpts.

Full UR3 example commands are in [UR3_CMD_README.txt](UR3_CMD_README.txt).

### 1.4 Open-loop evaluation (standalone)

Replay a checkpoint frame-by-frame over a given episode, comparing predicted action chunks against the demo ground truth, and write comparison plots to `out_dir`:

```bash
uv run scripts/eval.py --config pi05_unitree_g1 \
    --checkpoint checkpoints/pi05_unitree_g1/v1/16000 \
    --data_path /home/bioprocessing-lab/yuhao/data/put_cup_n_broccoli_eval \
    --episode 0 --stride 1 --out_dir eval_out
```

### 1.5 Start an inference server

```bash
# Standard server (recommended): automatically drops the actions<-action mapping
# from RepackTransform, so the client does not need to send ground-truth actions
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_unitree_g1 \
    --policy.dir=checkpoints/pi05_unitree_g1/v1/16000

# Server with request logging: saves each inference's obs and predicted chunk to --log_dir
uv run scripts/serve_policy_log.py policy:checkpoint \
    --policy.config=pi05_unitree_g1 \
    --policy.dir=checkpoints/pi05_unitree_g1/v1/16000 \
    --log_dir inference_logs
```

---

## 2. Changes Relative to Upstream

| File | Change |
| --- | --- |
| [src/openpi/policies/unitree_policy.py](src/openpi/policies/unitree_policy.py) | New. Unitree G1 Dex1 input/output transforms (16-DoF: L arm 7 + R arm 7 + L gripper 1 + R gripper 1; three cameras left_high/left_wrist/right_wrist). `_parse_image` supports JPEG byte decoding (works with the client's `--send-jpeg` compressed upload). |
| [src/openpi/policies/ur3_policy.py](src/openpi/policies/ur3_policy.py) | New. UR3 input/output transforms (7-DoF: 6 joints + 1 gripper; two cameras cam_high/cam_wrist, right-wrist camera zero-padded). |
| [src/openpi/training/config.py](src/openpi/training/config.py) | Added `DataConfig.local_root` (load datasets from local disk at `local_root/repo_id` instead of the HF Hub); added `eval_*` fields to `TrainConfig` (`eval_interval/eval_episodes/eval_stride/eval_max_frames/eval_data_path`); added data config classes `LeRobotUnitreeG1DataConfig`, `LeRobotUR3MergedDataConfig`, `OldLeRobotUR3MergedDataConfig`, `LeRobotUR3DataConfig`; added several `pi05_unitree_g1` / `pi05_ur3_*` training configs. |
| [src/openpi/training/data_loader.py](src/openpi/training/data_loader.py) | Passes `data_config.local_root` as `root` to LeRobotDataset / Metadata to enable local dataset loading. |
| [scripts/serve_policy.py](scripts/serve_policy.py) | When creating the serving policy, removes the `actions<-action` mapping from the repack structure (no ground-truth action at inference time). |
| [scripts/serve_policy_log.py](scripts/serve_policy_log.py) | New. Wraps the policy to log each inference's obs and predicted chunk to disk for debugging. |
| [scripts/serve_policy_gguf.py](scripts/serve_policy_gguf.py) | New. Adds a GGUF quantized-model inference server (OmniModel.cpp backend). |
| [scripts/eval.py](scripts/eval.py) | New. Frame-by-frame open-loop eval of a single checkpoint, producing pred-vs-truth comparison plots. |
| [scripts/train_eval.py](scripts/train_eval.py) | New. Adds periodic open-loop eval during training (on top of train.py), logged to wandb. |
| [scripts/split_dataset.py](scripts/split_dataset.py) | New. Splits a held-out eval set from a LeRobot v2.1 dataset and re-indexes it. |
| [src/openpi/policies/policy.py](src/openpi/policies/policy.py) | Attaches VRAM telemetry to inference results (JAX path only, fully guarded, never affects inference). |
| [packages/openpi-client/src/openpi_client/image_tools.py](packages/openpi-client/src/openpi_client/image_tools.py) | Adds `_maybe_decode_encoded` before `resize_with_pad` to auto-decode images sent as compressed bytes (JPEG/PNG). |

---

## 3. What to Modify Before Training

To train on your own data, go through this checklist (using G1 as the example, referencing the [pi05_unitree_g1](src/openpi/training/config.py#L748) config):

1. **Prepare the dataset**: Convert to LeRobot v2.1 format and make sure the column names match the policy's `REPACK_STRUCTURE` (G1: `observation.images.cam_left_high/cam_left_wrist/cam_right_wrist`, `observation.state`, `action`). If the names differ, edit `REPACK_STRUCTURE` in the policy file and the repack mapping in the config.

2. **Create/adjust the TrainConfig** (in the `_CONFIGS` list in [src/openpi/training/config.py](src/openpi/training/config.py)):
   - `data.repo_id`: local dataset directory name (relative to `local_root`) or HF Hub repo id.
   - `data.base_config.local_root`: local dataset root directory (set this for local data; omit to pull from the Hub).
   - `model`: `action_horizon`, `pi05`, LoRA variants, etc. (action dimension is set by the policy's `ACTION_DIM`: G1=16, UR3=7).
   - Hyperparameters: `batch_size`, `num_train_steps`, `save_interval`, `keep_period`, `lr_schedule`, `optimizer`, `fsdp_devices` (multi-GPU), `ema_decay`.
   - `weight_loader`: base checkpoint path (default `gs://openpi-assets/checkpoints/pi05_base/params`).
   - Eval fields: `eval_interval`, `eval_episodes`, `eval_stride`, `eval_data_path` (when using train_eval.py).

3. **If dimensions/cameras don't match, edit the policy transform**: in [unitree_policy.py](src/openpi/policies/unitree_policy.py) / [ur3_policy.py](src/openpi/policies/ur3_policy.py), adjust `STATE_DIM`/`ACTION_DIM`, the number of cameras, `image_mask`, and the output slice `actions[:, :ACTION_DIM]`.

4. **Recompute norm stats**: after changing the dataset or repack mapping, always rerun `compute_norm_stats.py`, otherwise training uses stale normalization statistics.

> Norm stats gotcha: using an absolute path for `repo_id` causes the checkpoint to fail to find its normalization statistics. Use `local_root` + a relative `repo_id` instead.

---

## Upstream Docs

### Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.


