cd /data/workspace/Codes/torchrec_based_projects/torchrec-dlrm

torchx run -s local_cwd dist.ddp -j 1x1 --script dlrm_pep/dlrm_kaggle.py -- \
    --dataset_name="criteo_kaggle" \
    --in_memory_binary_criteo_path="datasets/criteo_kaggle/torchrec_processed" \
    --seed=5828 \
    --epochs=1 \
    --batch_size=4096 \
    --mmap_mode \
    --adagrad \
    --learning_rate=0.006 \
    --limit_train_batches=200000 \
    --limit_val_batches=10 \
    --limit_test_batches=200 \
    --validation_freq_within_epoch=10 \
    --test_batch_size=8192
