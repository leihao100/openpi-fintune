"""Split a LeRobot (v2.1) dataset: copy randomly-selected episodes into an eval
folder as a valid, re-indexed dataset.

Example:
    python scripts/split_dataset.py \
        --src /home/bioprocessing-lab/yuhao/data/put_cup_n_broccoli \
        --eval_dir /home/bioprocessing-lab/yuhao/data/put_cup_n_broccoli_eval \
        --num 5 --seed 0

    uv run scripts/split_dataset.py \
        --src ~/unitree/data/sort-tools-eef \
        --eval_dir ~/unitree/data/sort-tools-eef-eval \
        --num 5 --seed 0  --move 

By default the source is left untouched (eval episodes still exist in train).
Pass --move to also remove the selected episodes from the source, re-indexing it
in place, so the eval set becomes a true held-out split.
"""

import dataclasses
import json
import os
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro


@dataclasses.dataclass
class Args:
    # Source LeRobot dataset directory.
    src: str
    # Output directory for the eval split.
    eval_dir: str
    # Number of episodes to sample for eval (ignored if --episodes is given).
    num: int = 5
    # Random seed for selection.
    seed: int = 0
    # Explicit episode indices to use for eval instead of random sampling.
    episodes: list[int] | None = None
    # Also remove the selected episodes from the source (true held-out split).
    move: bool = False


def _build_subset(src: pathlib.Path, dst: pathlib.Path, episodes: list[int]) -> None:
    """Write `episodes` (old indices) from `src` into `dst`, re-indexed to 0..K-1."""
    info = json.loads((src / "meta/info.json").read_text())
    chunks_size = info["chunks_size"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    data_tmpl, video_tmpl = info["data_path"], info["video_path"]

    eps = {json.loads(l)["episode_index"]: json.loads(l) for l in (src / "meta/episodes.jsonl").open()}
    stats = {json.loads(l)["episode_index"]: json.loads(l) for l in (src / "meta/episodes_stats.jsonl").open()}

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    new_eps, new_stats, running = [], [], 0
    for new, old in enumerate(episodes):
        chunk = new // chunks_size
        src_pq = src / data_tmpl.format(episode_chunk=old // chunks_size, episode_index=old)
        dst_pq = dst / data_tmpl.format(episode_chunk=chunk, episode_index=new)
        dst_pq.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_pq)
        n = table.num_rows
        table = table.set_column(
            table.schema.get_field_index("episode_index"), "episode_index", pa.array([new] * n, pa.int64())
        )
        table = table.set_column(
            table.schema.get_field_index("index"), "index", pa.array(np.arange(running, running + n), pa.int64())
        )
        pq.write_table(table, dst_pq)

        for vk in video_keys:
            s = src / video_tmpl.format(episode_chunk=old // chunks_size, video_key=vk, episode_index=old)
            d = dst / video_tmpl.format(episode_chunk=chunk, video_key=vk, episode_index=new)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

        new_eps.append({**eps[old], "episode_index": new})
        new_stats.append({**stats[old], "episode_index": new})
        running += n

    with (dst / "meta/episodes.jsonl").open("w") as f:
        f.writelines(json.dumps(e) + "\n" for e in new_eps)
    with (dst / "meta/episodes_stats.jsonl").open("w") as f:
        f.writelines(json.dumps(s) + "\n" for s in new_stats)
    shutil.copy2(src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")

    k = len(episodes)
    info = {
        **info,
        "total_episodes": k,
        "total_frames": running,
        "total_videos": k * len(video_keys),
        "total_chunks": (k - 1) // chunks_size + 1 if k else 0,
        "splits": {"train": f"0:{k}"},
    }
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))


def main(args: Args) -> None:
    src = pathlib.Path(args.src)
    eval_dir = pathlib.Path(args.eval_dir)
    if eval_dir.resolve() == src.resolve():
        raise ValueError("eval_dir must differ from src.")

    n_total = json.loads((src / "meta/info.json").read_text())["total_episodes"]
    if args.episodes is not None:
        selected = sorted(args.episodes)
    else:
        selected = sorted(np.random.default_rng(args.seed).choice(n_total, args.num, replace=False).tolist())
    print(f"Selected {len(selected)} eval episodes: {selected}")

    _build_subset(src, eval_dir, selected)
    print(f"Wrote eval split -> {eval_dir}")

    if args.move:
        kept = [e for e in range(n_total) if e not in set(selected)]
        tmp = src.parent / (src.name + "__tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        _build_subset(src, tmp, kept)
        shutil.rmtree(src)
        os.rename(tmp, src)
        print(f"Re-indexed source to {len(kept)} train episodes -> {src}")


if __name__ == "__main__":
    main(tyro.cli(Args))
