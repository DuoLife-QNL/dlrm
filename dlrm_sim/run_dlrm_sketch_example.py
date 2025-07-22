#!/usr/bin/env python3
"""
Example script to demonstrate DLRM Kaggle training with sketch embedding.
This script shows how to use the modified dlrm_kaggle.py with sketch embedding enabled.
"""

import os
import sys
import subprocess
import argparse

def run_dlrm_sketch_training(args):
    """Run DLRM training with sketch embedding."""
    
    # Base command
    cmd = [
        "python", "dlrm_kaggle.py",
        "--use_sketch_embedding",
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--embedding_dim", str(args.embedding_dim),
        "--num_embeddings", str(args.num_embeddings),
        "--learning_rate", str(args.learning_rate),
        "--limit_train_batches", str(args.limit_train_batches),
        "--limit_val_batches", str(args.limit_val_batches),
        "--limit_test_batches", str(args.limit_test_batches),
    ]
    
    # Add sketch embedding parameters
    cmd.extend([
        "--sketch_hot_size_ratio", str(args.sketch_hot_size_ratio),
        "--sketch_hash_size_ratio", str(args.sketch_hash_size_ratio),
        "--sketch_threshold", str(args.sketch_threshold),
        "--sketch_alpha", str(args.sketch_alpha),
        "--sketch_adjust_threshold", str(args.sketch_adjust_threshold),
    ])
    
    # Add C++ acceleration if requested
    if args.sketch_use_cpp:
        cmd.append("--sketch_use_cpp")
    
    # Add other optional parameters
    if args.adagrad:
        cmd.append("--adagrad")
    
    if args.print_lr:
        cmd.append("--print_lr")
    
    if args.print_sharding_plan:
        cmd.append("--print_sharding_plan")
    
    print("Running DLRM training with sketch embedding...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ DLRM training with sketch embedding completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ DLRM training failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n❌ dlrm_kaggle.py not found. Please make sure you're in the correct directory.")
        return False

def main():
    """Main function to run DLRM sketch training example."""
    parser = argparse.ArgumentParser(description="DLRM Kaggle training with sketch embedding example")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")  # Changed from 16 to 64
    parser.add_argument("--num_embeddings", type=int, default=1000, help="Number of embeddings per table")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    
    # Limit parameters for quick testing
    parser.add_argument("--limit_train_batches", type=int, default=5, help="Limit training batches")
    parser.add_argument("--limit_val_batches", type=int, default=2, help="Limit validation batches")
    parser.add_argument("--limit_test_batches", type=int, default=2, help="Limit test batches")
    
    # Sketch embedding parameters
    parser.add_argument("--sketch_hot_size_ratio", type=float, default=0.1, help="Hot table size ratio")
    parser.add_argument("--sketch_hash_size_ratio", type=float, default=0.3, help="Hash table size ratio")
    parser.add_argument("--sketch_threshold", type=int, default=1, help="Sketch threshold")
    parser.add_argument("--sketch_alpha", type=float, default=1.0000005, help="Sketch alpha")
    parser.add_argument("--sketch_adjust_threshold", type=int, default=1, help="Sketch adjust threshold")
    parser.add_argument("--sketch_use_cpp", action="store_true", default=True, help="Use C++ acceleration")
    
    # Other options
    parser.add_argument("--adagrad", action="store_true", help="Use Adagrad optimizer")
    parser.add_argument("--print_lr", action="store_true", help="Print learning rate")
    parser.add_argument("--print_sharding_plan", action="store_true", help="Print sharding plan")
    
    args = parser.parse_args()
    
    print("DLRM Kaggle Training with Sketch Embedding Example")
    print("=" * 60)
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Number of embeddings: {args.num_embeddings}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Sketch hot ratio: {args.sketch_hot_size_ratio}")
    print(f"Sketch hash ratio: {args.sketch_hash_size_ratio}")
    print(f"Sketch threshold: {args.sketch_threshold}")
    print(f"Sketch alpha: {args.sketch_alpha}")
    print(f"Sketch adjust threshold: {args.sketch_adjust_threshold}")
    print(f"Use C++ acceleration: {args.sketch_use_cpp}")
    print(f"Use Adagrad: {args.adagrad}")
    print("-" * 60)
    
    # Run the training
    success = run_dlrm_sketch_training(args)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("\nKey features demonstrated:")
        print("✅ Sketch embedding compression")
        print("✅ Dynamic hot/cold feature management")
        print("✅ C++ acceleration for sketch operations")
        print("✅ TorchRec compatibility")
        print("✅ Distributed training support")
    else:
        print("\n⚠️  Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 