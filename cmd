uv run python -u train_torchrl.py \
  --num-envs 16 \
  --rollout-steps 4096 \
  --batch-size 512 \
  --ppo-epochs 10 \
  --compile \
  --video-interval 100000 \
  --total-steps 300_000_000
