#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../../config.sh"

if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file not found at $CONFIG_FILE"
    echo "Please make sure config.sh exists in the dlrm root directory"
    exit 1
fi

# Change to the configured working directory
cd "$DLRM_WORKING_DIR"

# Run DLRM Kaggle training with sketch embedding
torchx run -s local_cwd dist.ddp -j 1x2 --script dlrm_sim/dlrm_kaggle.py -- \
    --dataset_name="criteo_kaggle" \
    --in_memory_binary_criteo_path="datasets/criteo_kaggle/torchrec_processed" \
    --seed=5828 \
    --epochs=1 \
    --batch_size=4096 \
    --mmap_mode \
    --adagrad \
    --learning_rate=0.006 \
    --limit_train_batches=200000 \
    --limit_val_batches=100 \
    --limit_test_batches=2000 \
    --validation_freq_within_epoch=1000 \
    --test_batch_size=8192 \
    --use_sketch_embedding \
    --sketch_hot_size_ratio=0.1 \
    --sketch_hash_size_ratio=0.3 \
    --sketch_threshold=1 \
    --sketch_alpha=1.0000005 \
    --sketch_adjust_threshold=1 \
    --sketch_use_cpp 