# DLRM Kaggle Training with Sketch Embedding

This document describes how to use sketch embedding compression with DLRM Kaggle training.

## Overview

Sketch embedding is a memory-efficient embedding compression technique that dynamically manages hot and cold features using a count sketch data structure. It maintains separate embedding tables for frequently accessed (hot) and infrequently accessed (cold) features, automatically promoting features based on access frequency.

## Features

- **Memory Efficiency**: Reduces memory usage through dynamic hot/cold feature management
- **C++ Acceleration**: High-performance C++ implementation for sketch operations
- **TorchRec Compatibility**: Fully compatible with TorchRec's distributed training framework
- **Dynamic Adaptation**: Automatically adapts to changing feature access patterns
- **Configurable Compression**: Adjustable compression ratios and sketch parameters

## Installation

Make sure you have the sketch embedding module installed:

```bash
# The sketch embedding module should be in the cafe/ directory
# Make sure the C++ library is compiled
cd cafe/
python build.py
```

## Usage

### Basic Usage

To enable sketch embedding in DLRM Kaggle training, add the `--use_sketch_embedding` flag:

```bash
python dlrm_kaggle.py --use_sketch_embedding --batch_size 32 --epochs 1
```

### Advanced Configuration

You can customize sketch embedding parameters:

```bash
python dlrm_kaggle.py \
    --use_sketch_embedding \
    --batch_size 32 \
    --epochs 1 \
    --embedding_dim 64 \
    --num_embeddings 100000 \
    --sketch_hot_size_ratio 0.1 \
    --sketch_hash_size_ratio 0.3 \
    --sketch_threshold 1 \
    --sketch_alpha 1.0000005 \
    --sketch_adjust_threshold 1 \
    --sketch_use_cpp \
    --learning_rate 0.1
```

### Example Script

Use the provided example script for quick testing:

```bash
python run_dlrm_sketch_example.py --batch_size 32 --epochs 1
```

## Parameters

### Sketch Embedding Parameters

- `--use_sketch_embedding`: Enable sketch embedding compression
- `--sketch_hot_size_ratio`: Ratio of hot table size to total embeddings (default: 0.1)
- `--sketch_hash_size_ratio`: Ratio of hash table size to total embeddings (default: 0.3)
- `--sketch_threshold`: Threshold for sketch data structure (default: 1)
- `--sketch_alpha`: Alpha parameter for sketch decay (default: 1.0000005)
- `--sketch_adjust_threshold`: Adjustment threshold for sketch reset (default: 1)
- `--sketch_use_cpp`: Enable C++ acceleration for sketch operations (default: True)

### Standard DLRM Parameters

All standard DLRM Kaggle parameters are supported:

- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--embedding_dim`: Embedding dimension
- `--num_embeddings`: Number of embeddings per table
- `--learning_rate`: Learning rate
- `--adagrad`: Use Adagrad optimizer
- `--interaction_type`: Interaction type (original, dcn, projection)

## Memory Savings

Sketch embedding provides significant memory savings:

- **Hot Table**: Stores frequently accessed features (typically 10% of total embeddings)
- **Hash Table**: Stores cold features with hash-based compression (typically 30% of total embeddings)
- **Total Memory**: ~40% of standard embedding table size

Example memory comparison:
- Standard embedding: 100,000 embeddings × 64 dims × 4 bytes = 25.6 MB
- Sketch embedding: (10,000 + 30,000) embeddings × 64 dims × 4 bytes = 10.2 MB
- **Memory savings**: ~60%

## Performance

### C++ Acceleration

The sketch embedding implementation includes C++ acceleration for:
- Sketch data structure operations
- Hot/cold feature detection
- Dynamic promotion/demotion

### Training Performance

Sketch embedding maintains training performance while reducing memory usage:
- Forward pass: Similar to standard embedding bags
- Backward pass: Optimized for sketch operations
- Memory access: Reduced due to smaller table sizes

## Testing

Run the test script to verify sketch embedding functionality:

```bash
python test_dlrm_sketch.py
```

This will test:
- Sketch embedding module imports
- DLRM Kaggle script integration
- Sketch embedding functionality
- Parameter parsing

## Troubleshooting

### Common Issues

1. **C++ Library Not Found**
   ```
   Warning: Failed to load C++ library: libsketch.so
   ```
   Solution: Compile the C++ library:
   ```bash
   cd cafe/
   python build.py
   ```

2. **Import Errors**
   ```
   ImportError: Sketch embedding is not available
   ```
   Solution: Make sure the sketch embedding modules are in the correct location.

3. **Type Errors**
   ```
   TypeError: 'float' object cannot be interpreted as an integer
   ```
   Solution: Ensure `sketch_threshold` and `sketch_adjust_threshold` are integers.

### Debug Mode

Enable debug output by setting environment variables:
```bash
export TORCH_REC_DEBUG=1
python dlrm_kaggle.py --use_sketch_embedding
```

## Advanced Usage

### Custom Sketch Parameters

For different datasets, you may need to tune sketch parameters:

```bash
# For high-frequency features
python dlrm_kaggle.py \
    --use_sketch_embedding \
    --sketch_hot_size_ratio 0.2 \
    --sketch_hash_size_ratio 0.4 \
    --sketch_threshold 2

# For low-frequency features
python dlrm_kaggle.py \
    --use_sketch_embedding \
    --sketch_hot_size_ratio 0.05 \
    --sketch_hash_size_ratio 0.2 \
    --sketch_threshold 1
```

### Distributed Training

Sketch embedding supports distributed training with TorchRec:

```bash
torchrun --nproc_per_node=4 dlrm_kaggle.py \
    --use_sketch_embedding \
    --batch_size 128 \
    --epochs 10
```

## Monitoring

During training, you can monitor sketch embedding behavior:

- **Hot/Cold Ratio**: Check the ratio of features in hot vs hash tables
- **Promotion Rate**: Monitor how often features are promoted to hot table
- **Memory Usage**: Compare memory usage with standard embedding bags

## References

- CAFE: Compressing Aggregated Features for Deep Learning Recommendation Models
- TorchRec: A PyTorch Library for Recommendation Systems
- DLRM: Deep Learning Recommendation Model

## License

This implementation is part of the TorchRec project and follows the same license terms. 