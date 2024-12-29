docker run -d --rm --runtime=nvidia --gpus device=1 \
  -v /mnt/datasets/datasets:/datasets -v ~/HiDe-Prompt:/workspace \
  -v ~/HiDe-Prompt/checkpoints:/checkpoints -p 3334:22 \
  --shm-size 32G hide:1.0 \
bash training_scripts/train_cgqa50-10_vit.sh