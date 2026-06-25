"""Same as scripts/train.py, but runs periodic open-loop eval during training
and logs action MAE/MSE + a demo-vs-pred plot to wandb.

Eval is controlled by the `eval_*` fields on TrainConfig (e.g. `--eval_interval 1000`).
It replays whole episodes from the (training or held-out) dataset through the
*live* model and compares predicted action chunks against the demo actions.
"""

import dataclasses
import functools
import gc
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pathlib
import torch
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


# --------------------------------------------------------------------------- #
# Open-loop eval.
# --------------------------------------------------------------------------- #


def _to_obs(frame: dict) -> dict:
    """Convert a raw LeRobot frame to numpy, leaving non-tensor values as-is."""
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in frame.items()}


def setup_eval(config: _config.TrainConfig, data_config: _config.DataConfig):
    """Load the eval dataset once and resolve the per-episode frame ranges."""
    action_horizon = config.model.action_horizon
    action_key = data_config.action_sequence_keys[0]

    if config.eval_data_path is not None:
        root = pathlib.Path(config.eval_data_path)
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

    specs = []
    for e in range(config.eval_episodes):
        start = int(dataset.episode_data_index["from"][e])
        end = int(dataset.episode_data_index["to"][e])
        idxs = list(range(start, end, config.eval_stride))
        if config.eval_max_frames is not None:
            idxs = idxs[: config.eval_max_frames]
        specs.append((e, start, idxs))
    return dataset, prompt_fn, action_key, specs


def _build_eval_policy(state: training_utils.TrainState, data_config: _config.DataConfig) -> _policy.Policy:
    """Wrap the live in-training model with the same transforms used at serving time."""
    use_ema = state.ema_decay is not None and state.ema_params is not None
    model = nnx.merge(state.model_def, state.ema_params if use_ema else state.params)
    model.eval()
    norm_stats = data_config.norm_stats
    return _policy.Policy(
        model,
        transforms=[
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *data_config.repack_transforms.outputs,
        ],
    )


def _actions_fig(demo: np.ndarray, pred: np.ndarray, t: np.ndarray, name: str, episode: int):
    """Per-dim executed action (chunk step 0): demo vs predicted."""
    d = demo.shape[-1]
    ncols = min(4, d)
    nrows = -(-d // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for j in range(d):
        ax = axes[j // ncols][j % ncols]
        ax.plot(t, demo[:, 0, j], label="demo", color="tab:blue")
        ax.plot(t, pred[:, 0, j], label="pred", color="tab:orange", linestyle="--")
        ax.set_title(f"dim {j}  (MAE {np.abs(pred - demo)[:, 0, j].mean():.4f})")
        ax.set_xlabel("frame")
    axes[0][0].legend()
    for k in range(d, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{name}  episode {episode}  —  executed action (chunk step 0)")
    fig.tight_layout()
    return fig


def run_eval(config, data_config, state, dataset, prompt_fn, action_key, specs, mesh):
    """Replay episodes through the live model and return wandb metrics + figures."""
    policy = _build_eval_policy(state, data_config)
    metrics: dict[str, Any] = {}
    sq_sum = abs_sum = count = 0.0
    with sharding.set_mesh(mesh):
        for e, start, idxs in specs:
            demo, pred = [], []
            for idx in idxs:
                frame = dataset[idx]
                if prompt_fn is not None:
                    frame = prompt_fn(frame)
                out = policy.infer(_to_obs(frame))
                demo.append(np.asarray(frame[action_key]))
                pred.append(np.asarray(out["actions"]))
            demo, pred = np.stack(demo), np.stack(pred)
            d = min(demo.shape[-1], pred.shape[-1])
            demo, pred = demo[..., :d], pred[..., :d]
            err = pred - demo
            metrics[f"eval/ep{e}_mae"] = float(np.abs(err).mean())
            abs_sum += np.abs(err).sum()
            sq_sum += np.square(err).sum()
            count += err.size
            if e == specs[0][0]:
                metrics["eval/actions"] = wandb.Image(
                    _actions_fig(demo, pred, np.asarray(idxs) - start, config.name, e)
                )
    metrics["eval/mae"] = float(abs_sum / count)
    metrics["eval/mse"] = float(sq_sum / count)
    del policy
    plt.close("all")
    gc.collect()
    return metrics


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Set up eval (loads the eval dataset once).
    data_config = data_loader.data_config()
    eval_state = None
    if config.eval_interval:
        eval_state = setup_eval(config, data_config)
        logging.info(
            "Eval enabled: every %d steps on %d episode(s)", config.eval_interval, config.eval_episodes
        )

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if eval_state is not None and step % config.eval_interval == 0 and step > start_step:
            eval_metrics = run_eval(config, data_config, train_state, *eval_state, mesh)
            scalars = {k: v for k, v in eval_metrics.items() if not isinstance(v, wandb.Image)}
            pbar.write(f"Step {step}: eval " + ", ".join(f"{k}={v:.4f}" for k, v in scalars.items()))
            wandb.log(eval_metrics, step=step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
