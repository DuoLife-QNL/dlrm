cd /data/workspace/Codes/torchrec_based_projects/torchrec-dlrm

# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client

torchx run -s local_cwd dist.ddp -j 1x1 -m debugpy -- --listen 0.0.0.0:5678 --wait-for-client dlrm_pep/dlrm_kaggle.py \
    --dataset_name="criteo_kaggle" \
    --in_memory_binary_criteo_path="datasets/criteo_kaggle/torchrec_processed" \
    --seed=5828 \
    --mmap_mode \
    --batch_size=16 \
    --limit_train_batches=200 \
    --limit_val_batches=200 \
    --print_lr \
    --test_batch_size=8192
