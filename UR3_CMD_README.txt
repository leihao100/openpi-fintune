 Command to compute norm
 uv run scripts/compute_norm_stats.py --config-name pi05_ur3_5task
 
 Command to infer in server
 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_ur3_5task --policy.dir=/home/ur3-exp/pi/openpi/checkpoints/pi05_ur3_multitask_full_action/v3/6000

Command to train

uv run scripts/train.py pi05_ur3_5task     --exp-name=v3
