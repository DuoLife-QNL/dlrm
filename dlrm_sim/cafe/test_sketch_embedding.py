#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Dict, List

# Import TorchRec modules
from torchrec.modules.embedding_configs import DataType, PoolingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor

# Import sketch modules
from sketch_embedding_config import SketchEmbeddingBagConfig
from sketch_embedding_bag import SKEmbeddingBag
from sketch_embedding_collection import SketchEmbeddingBagCollection


def create_sketch_embedding_bag_example():
    """
    Example demonstrating how to use Sketch-based EmbeddingBag.
    
    This shows the CAFE sketch-based embedding compression technique
    integrated with TorchRec's framework.
    """
    
    # Define sketch embedding table configuration
    config = SketchEmbeddingBagConfig(
        name="sketch_table",
        embedding_dim=16,
        num_embeddings=10000,  # Total vocabulary size
        feature_names=["user_id"],
        data_type=DataType.FP32,
        pooling=PoolingType.SUM,
        # Sketch-specific parameters
        hot_size=1000,  # Size of hot embedding table
        hash_size=5000,  # Size of hash embedding table
        compress_rate=0.1,  # 10% compression rate
        sketch_threshold=1,
        sketch_alpha=1.0000005,
        hash_rate=0.3,
        adjust_threshold=1,
    )
    
    # Create SKEmbeddingBag
    sketch_embedding_bag = SKEmbeddingBag(
        config=config,
        device=torch.device("cpu"),
    )
    
    print("SKEmbeddingBag created successfully!")
    print(f"Embedding dimension: {sketch_embedding_bag.embedding_dim}")
    print(f"Hot table size: {sketch_embedding_bag.hot_size}")
    print(f"Hash table size: {sketch_embedding_bag.hash_size}")
    print(f"Total embeddings: {sketch_embedding_bag.num_embeddings}")
    
    return sketch_embedding_bag


def create_sketch_embedding_collection_example():
    """
    Example demonstrating how to use SketchEmbeddingBagCollection.
    """
    
    # Define multiple sketch embedding table configurations
    table_configs = [
        SketchEmbeddingBagConfig(
            name="user_table",
            embedding_dim=16,
            num_embeddings=10000,
            feature_names=["user_id"],
            data_type=DataType.FP32,
            pooling=PoolingType.SUM,
            hot_size=1000,
            hash_size=5000,
            compress_rate=0.1,
        ),
        SketchEmbeddingBagConfig(
            name="item_table",
            embedding_dim=16,
            num_embeddings=5000,
            feature_names=["item_id"],
            data_type=DataType.FP32,
            pooling=PoolingType.SUM,
            hot_size=500,
            hash_size=2500,
            compress_rate=0.1,
        ),
        SketchEmbeddingBagConfig(
            name="category_table",
            embedding_dim=16,
            num_embeddings=1000,
            feature_names=["category_id"],
            data_type=DataType.FP32,
            pooling=PoolingType.SUM,
            hot_size=100,
            hash_size=500,
            compress_rate=0.1,
        ),
    ]
    
    # Create SketchEmbeddingBagCollection
    sketch_collection = SketchEmbeddingBagCollection(
        tables=table_configs,
        device=torch.device("cpu"),
    )
    
    print("SketchEmbeddingBagCollection created successfully!")
    print(f"Number of tables: {len(sketch_collection.embedding_bag_configs())}")
    print(f"Feature names: {sketch_collection._embedding_names}")
    
    return sketch_collection


def create_sample_data():
    """
    Create sample sparse data for demonstration.
    
    Returns:
        KeyedJaggedTensor: Sample sparse features
    """
    
    # Sample data: 3 batches, each with different sparse features
    # user_id: [0,1], None, [2] for batches 0,1,2
    # item_id: [3], [4], [5,6,7] for batches 0,1,2  
    # category_id: [8], [9], [10] for batches 0,1,2
    
    features = KeyedJaggedTensor.from_offsets_sync(
        keys=["user_id", "item_id", "category_id"],
        values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8, 9, 10, 11]),
    )
    
    print("Sample data created:")
    print(f"Keys: {features.keys()}")
    print(f"Values: {features.values()}")
    print(f"Offsets: {features.offsets()}")
    
    return features


def test_sketch_embedding_bag(sketch_embedding_bag: SKEmbeddingBag):
    """
    Test individual SKEmbeddingBag functionality.
    
    Args:
        sketch_embedding_bag: The SKEmbeddingBag instance
    """
    
    print("\n" + "="*50)
    print("SKETCH EMBEDDING BAG TEST")
    print("="*50)
    
    # Create sample input
    input_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    offsets = torch.tensor([0, 2, 5, 8, 11])
    
    print(f"Input indices: {input_indices}")
    print(f"Offsets: {offsets}")
    
    # Test training mode
    sketch_embedding_bag.train()
    print("\n--- Training Mode ---")
    
    # Forward pass
    output = sketch_embedding_bag(input_indices, offsets)
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[:3]}")
    
    # Test inference mode
    sketch_embedding_bag.eval()
    print("\n--- Inference Mode ---")
    
    # Forward pass
    output = sketch_embedding_bag(input_indices, offsets)
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[:3]}")
    
    # Test gradient flow
    print("\n--- Gradient Flow Test ---")
    sketch_embedding_bag.train()
    output = sketch_embedding_bag(input_indices, offsets)
    
    # Create a simple loss
    loss = output.sum()
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    if sketch_embedding_bag.weight_hot.grad is not None:
        hot_grad_norm = sketch_embedding_bag.weight_hot.grad.norm().item()
        print(f"Hot table gradient norm: {hot_grad_norm:.4f}")
    
    if sketch_embedding_bag.weight_hash.grad is not None:
        hash_grad_norm = sketch_embedding_bag.weight_hash.grad.norm().item()
        print(f"Hash table gradient norm: {hash_grad_norm:.4f}")
    
    # Test sketch info
    sketch_info = sketch_embedding_bag.get_sketch_info()
    print(f"\nSketch info: {sketch_info}")


def test_sketch_embedding_collection(sketch_collection: SketchEmbeddingBagCollection, features: KeyedJaggedTensor):
    """
    Test SketchEmbeddingBagCollection functionality.
    
    Args:
        sketch_collection: The SketchEmbeddingBagCollection instance
        features: Input sparse features
    """
    
    print("\n" + "="*50)
    print("SKETCH EMBEDDING COLLECTION TEST")
    print("="*50)
    
    # Test training mode
    sketch_collection.train()
    print("\n--- Training Mode ---")
    
    # Forward pass
    output = sketch_collection(features)
    print(f"Output keys: {list(output.keys())}")
    print(f"Output shape: {output.values().shape}")
    print(f"Output sample: {output.values()[:3]}")
    
    # Test inference mode
    sketch_collection.eval()
    print("\n--- Inference Mode ---")
    
    # Forward pass
    output = sketch_collection(features)
    print(f"Output keys: {list(output.keys())}")
    print(f"Output shape: {output.values().shape}")
    print(f"Output sample: {output.values()[:3]}")
    
    # Test gradient flow
    print("\n--- Gradient Flow Test ---")
    sketch_collection.train()
    output = sketch_collection(features)
    
    # Create a simple loss
    loss = output.values().sum()
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients for each table
    for i, embedding_bag in enumerate(sketch_collection.embedding_bags):
        table_name = sketch_collection._embedding_names[i]
        if embedding_bag.weight_hot.grad is not None:
            hot_grad_norm = embedding_bag.weight_hot.grad.norm().item()
            print(f"{table_name} hot table gradient norm: {hot_grad_norm:.4f}")
        
        if embedding_bag.weight_hash.grad is not None:
            hash_grad_norm = embedding_bag.weight_hash.grad.norm().item()
            print(f"{table_name} hash table gradient norm: {hash_grad_norm:.4f}")
    
    # Test sketch stats
    sketch_stats = sketch_collection.get_sketch_stats()
    print(f"\nSketch stats: {sketch_stats}")


def test_torchrec_compatibility(use_cpp=False):
    """Test compatibility with TorchRec interfaces."""
    print("\n" + "="*50)
    print("TORCHREC COMPATIBILITY TEST")
    print("="*50)
    
    # Create a simple model using sketch embedding
    class SimpleModel(nn.Module):
        def __init__(self, use_cpp=False):
            super().__init__()
            config = SketchEmbeddingBagConfig(
                name="test_table",
                embedding_dim=16,
                num_embeddings=1000,
                feature_names=["test_feature"],
                hot_size=100,
                hash_size=500
            )
            self.embedding = SKEmbeddingBag(config=config)
            self.linear = nn.Linear(16, 1)
            
        def forward(self, indices, offsets):
            embedded = self.embedding(indices, offsets)
            return self.linear(embedded)
    
    model = SimpleModel(use_cpp=use_cpp)
    print("Testing model integration...")
    
    # Test forward pass
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    offsets = torch.tensor([0, 2, 4, 6, 8])
    
    model.train()
    output = model(indices, offsets)
    print(f"Model output shape: {output.shape}")
    print(f"Model output: {output}")
    
    # Test training
    target = torch.randn(output.shape)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    print("Model training successful!")
    print("✅ TorchRec compatibility test passed!")


def main():
    """
    Main function demonstrating sketch embedding usage.
    """
    
    print("Sketch Embedding Test")
    print("="*50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test SKEmbeddingBag
    print("1. Testing SKEmbeddingBag...")
    config = SketchEmbeddingBagConfig(
        name="test_table",
        embedding_dim=16,
        num_embeddings=10000,
        feature_names=["test_feature"],
        hot_size=1000,
        hash_size=5000
    )
    embedding_bag = SKEmbeddingBag(config=config, use_cpp=True)
    test_sketch_embedding_bag(embedding_bag)
    
    # Test SketchEmbeddingBagCollection
    print("\n2. Testing SketchEmbeddingBagCollection...")
    configs = [
        SketchEmbeddingBagConfig(
            name="user_id",
            embedding_dim=16,
            num_embeddings=10000,
            feature_names=["user_id"],
            hot_size=1000,
            hash_size=5000
        ),
        SketchEmbeddingBagConfig(
            name="item_id", 
            embedding_dim=16,
            num_embeddings=5000,
            feature_names=["item_id"],
            hot_size=500,
            hash_size=2500
        ),
        SketchEmbeddingBagConfig(
            name="category_id",
            embedding_dim=16, 
            num_embeddings=1000,
            feature_names=["category_id"],
            hot_size=100,
            hash_size=500
        )
    ]
    embedding_collection = SketchEmbeddingBagCollection(tables=configs, use_cpp=True)
    features = create_sample_data()
    test_sketch_embedding_collection(embedding_collection, features)
    
    # Test TorchRec compatibility
    print("\n3. Testing TorchRec compatibility...")
    test_torchrec_compatibility(use_cpp=True)  # Enable C++ acceleration
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    main() 