torchx run -s local_cwd dist.ddp -j 1x1 --script ../dlrm_kaggle.py -- \
    --dataset_name="criteo_kaggle" \
    --in_memory_binary_criteo_path="../../datasets/criteo_kaggle/torchrec_processed" \
    --seed=5828 \
    --mmap_mode \
    --limit_train_batches=200 \
    --limit_test_batches=200
