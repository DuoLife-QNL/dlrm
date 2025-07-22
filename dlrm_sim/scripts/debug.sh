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

# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client

torchx run -s local_cwd dist.ddp -j 1x1 -m debugpy  -- --listen 0.0.0.0:5678 --wait-for-client dlrm_sim/dlrm_kaggle.py \
    --dataset_name="criteo_kaggle" \
    --in_memory_binary_criteo_path="datasets/criteo_kaggle/torchrec_processed" \
    --seed=5828 \
    --mmap_mode \
    --batch_size=16 \
    --limit_train_batches=200 \
    --limit_val_batches=200 \
    --print_lr \
    --test_batch_size=8192