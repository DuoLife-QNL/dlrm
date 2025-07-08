#!/bin/bash

cd /data/workspace/Codes/torchrec_based_projects/torchrec-dlrm

torchx run -s local_cwd dist.ddp -j 1x1 --script dlrm_pep/dlrm_kaggle.py -- \
    --dataset_name="criteo_kaggle" \
    --in_memory_binary_criteo_path="datasets/criteo_kaggle/torchrec_processed" \
    --seed=5828 \
    --epochs=3 \
    --batch_size=1024 \
    --test_batch_size=2048 \
    --mmap_mode \
    --adagrad \
    --learning_rate=0.01 \
    --eps=1e-8 \
    --limit_train_batches=50000 \
    --limit_val_batches=1000 \
    --limit_test_batches=1000 \
    --validation_freq_within_epoch=1000 \
    --dense_arch_layer_sizes="256,128,64" \
    --over_arch_layer_sizes="128,64,1" \
    --embedding_dim=64 \
    --lr_warmup_steps=100 \
    --lr_decay_start=1000 \
    --lr_decay_steps=2000 