#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Dict, List

# Import TorchRec modules
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor


def create_embedding_collection_example():
    """
    Example demonstrating how to use TorchRec's EmbeddingCollection.
    
    EmbeddingCollection represents a collection of non-pooled embeddings.
    It processes sparse data and outputs individual embeddings for each feature.
    """
    
    # Define embedding table configurations
    table_configs = [
        EmbeddingConfig(
            name="user_table",
            embedding_dim=16,
            num_embeddings=1000,
            feature_names=["user_id"],
            data_type=DataType.FP32,
        ),
        EmbeddingConfig(
            name="item_table", 
            embedding_dim=16,
            num_embeddings=500,
            feature_names=["item_id"],
            data_type=DataType.FP32,
        ),
        EmbeddingConfig(
            name="category_table",
            embedding_dim=16, 
            num_embeddings=100,
            feature_names=["category_id"],
            data_type=DataType.FP32,
        ),
    ]
    
    # Create EmbeddingCollection
    embedding_collection = EmbeddingCollection(
        tables=table_configs,
        device=torch.device("cpu"),
        need_indices=False,
    )
    
    print("EmbeddingCollection created successfully!")
    print(f"Embedding dimension: {embedding_collection.embedding_dim()}")
    print(f"Number of tables: {len(embedding_collection.embedding_configs())}")
    print(f"Feature names by table: {embedding_collection.embedding_names_by_table()}")
    
    return embedding_collection


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


def demonstrate_forward_pass(embedding_collection: EmbeddingCollection, features: KeyedJaggedTensor):
    """
    Demonstrate the forward pass of EmbeddingCollection.
    
    Args:
        embedding_collection: The EmbeddingCollection instance
        features: Input sparse features
    """
    
    print("\n" + "="*50)
    print("FORWARD PASS DEMONSTRATION")
    print("="*50)
    
    # Forward pass
    feature_embeddings: Dict[str, JaggedTensor] = embedding_collection(features)
    
    print(f"Output keys: {list(feature_embeddings.keys())}")
    
    # Examine embeddings for each feature
    for feature_name, jagged_tensor in feature_embeddings.items():
        print(f"\nFeature: {feature_name}")
        print(f"  Embedding shape: {jagged_tensor.values().shape}")
        print(f"  Lengths: {jagged_tensor.lengths()}")
        print(f"  Sample embeddings:")
        print(f"    {jagged_tensor.values()[:3]}")  # Show first 3 embeddings
        
        # Calculate statistics
        embeddings = jagged_tensor.values()
        print(f"  Statistics:")
        print(f"    Mean: {embeddings.mean():.4f}")
        print(f"    Std: {embeddings.std():.4f}")
        print(f"    Min: {embeddings.min():.4f}")
        print(f"    Max: {embeddings.max():.4f}")


def demonstrate_gradient_flow(embedding_collection: EmbeddingCollection, features: KeyedJaggedTensor):
    """
    Demonstrate gradient flow through the EmbeddingCollection.
    
    Args:
        embedding_collection: The EmbeddingCollection instance
        features: Input sparse features
    """
    
    print("\n" + "="*50)
    print("GRADIENT FLOW DEMONSTRATION")
    print("="*50)
    
    # Forward pass
    feature_embeddings = embedding_collection(features)
    
    # Create a simple loss (sum of all embeddings)
    total_loss = torch.tensor(0.0, requires_grad=True)
    for feature_name, jagged_tensor in feature_embeddings.items():
        total_loss = total_loss + jagged_tensor.values().sum()
    
    print(f"Initial loss: {total_loss.item():.4f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients for each embedding table
    for table_name, embedding_module in embedding_collection.embeddings.items():
        if embedding_module.weight.grad is not None:
            grad_norm = embedding_module.weight.grad.norm().item()
            print(f"Gradient norm for {table_name}: {grad_norm:.4f}")
        else:
            print(f"No gradient for {table_name}")


def demonstrate_parameter_reset(embedding_collection: EmbeddingCollection):
    """
    Demonstrate parameter reset functionality.
    
    Args:
        embedding_collection: The EmbeddingCollection instance
    """
    
    print("\n" + "="*50)
    print("PARAMETER RESET DEMONSTRATION")
    print("="*50)
    
    # Store original parameters
    original_params = {}
    for table_name, embedding_module in embedding_collection.embeddings.items():
        original_params[table_name] = embedding_module.weight.clone()
    
    print("Original parameter statistics:")
    for table_name, param in original_params.items():
        print(f"  {table_name}: mean={param.mean():.4f}, std={param.std():.4f}")
    
    # Reset parameters
    embedding_collection.reset_parameters()
    
    print("\nAfter reset parameter statistics:")
    for table_name, embedding_module in embedding_collection.embeddings.items():
        param = embedding_module.weight
        print(f"  {table_name}: mean={param.mean():.4f}, std={param.std():.4f}")
    
    # Check if parameters changed
    print("\nParameter change check:")
    for table_name, embedding_module in embedding_collection.embeddings.items():
        param_diff = (embedding_module.weight - original_params[table_name]).abs().max().item()
        print(f"  {table_name}: max difference = {param_diff:.6f}")


def main():
    """
    Main function demonstrating EmbeddingCollection usage.
    """
    
    print("TorchRec EmbeddingCollection Example")
    print("="*50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create embedding collection
    embedding_collection = create_embedding_collection_example()
    
    # Create sample data
    features = create_sample_data()
    
    # Demonstrate forward pass
    demonstrate_forward_pass(embedding_collection, features)
    
    # Demonstrate gradient flow
    demonstrate_gradient_flow(embedding_collection, features)
    
    # Demonstrate parameter reset
    demonstrate_parameter_reset(embedding_collection)
    
    print("\n" + "="*50)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    main()
