#!/usr/bin/env python3
"""
Test script to verify DLRM Kaggle training with sketch embedding.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dlrm_sketch_import():
    """Test if sketch embedding modules can be imported."""
    try:
        from cafe.sketch_embedding_bag import SKEmbeddingBag
        from cafe.sketch_embedding_collection import SketchEmbeddingBagCollection
        from cafe.sketch_embedding_config import SketchEmbeddingBagConfig
        print("✅ Sketch embedding modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import sketch embedding modules: {e}")
        return False

def test_dlrm_kaggle_import():
    """Test if DLRM Kaggle script can be imported."""
    try:
        from dlrm_kaggle import parse_args, main
        print("✅ DLRM Kaggle script imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import DLRM Kaggle script: {e}")
        return False

def test_sketch_embedding_functionality():
    """Test basic sketch embedding functionality."""
    try:
        from cafe.sketch_embedding_config import SketchEmbeddingBagConfig
        from cafe.sketch_embedding_bag import SKEmbeddingBag
        
        # Create a simple sketch embedding config
        config = SketchEmbeddingBagConfig(
            name="test_table",
            embedding_dim=16,
            num_embeddings=1000,
            feature_names=["test_feature"],
            hot_size=100,
            hash_size=300,
            sketch_threshold=1,
            sketch_alpha=1.0000005,
            adjust_threshold=1,  # Changed from 0.5 to 1 (int)
        )
        
        # Create sketch embedding bag
        embedding_bag = SKEmbeddingBag(config=config, use_cpp=True)
        
        # Test forward pass
        indices = torch.tensor([0, 1, 2, 3, 4, 5])
        offsets = torch.tensor([0, 2, 4, 6])
        
        embedding_bag.train()
        output = embedding_bag(indices, offsets)
        
        print(f"✅ Sketch embedding forward pass successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Sketch embedding functionality test failed: {e}")
        return False

def test_dlrm_with_sketch_args():
    """Test DLRM with sketch embedding arguments."""
    try:
        from dlrm_kaggle import parse_args
        
        # Create test arguments with sketch embedding enabled
        test_args = [
            "--use_sketch_embedding",
            "--sketch_hot_size_ratio", "0.1",
            "--sketch_hash_size_ratio", "0.3",
            "--sketch_threshold", "1",
            "--sketch_alpha", "1.0000005",
            "--sketch_adjust_threshold", "1",  # Changed from "0.5" to "1"
            "--sketch_use_cpp",
            "--batch_size", "32",
            "--epochs", "1",
            "--limit_train_batches", "2",
            "--limit_val_batches", "1",
            "--limit_test_batches", "1",
            "--embedding_dim", "16",
            "--num_embeddings", "1000",
            "--learning_rate", "0.1",
        ]
        
        args = parse_args(test_args)
        
        # Check if sketch embedding arguments are parsed correctly
        assert args.use_sketch_embedding == True
        assert args.sketch_hot_size_ratio == 0.1
        assert args.sketch_hash_size_ratio == 0.3
        assert args.sketch_threshold == 1
        assert args.sketch_alpha == 1.0000005
        assert args.sketch_adjust_threshold == 1
        assert args.sketch_use_cpp == True
        
        print("✅ DLRM sketch embedding arguments parsed correctly")
        return True
    except Exception as e:
        print(f"❌ DLRM sketch embedding arguments test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing DLRM Kaggle with Sketch Embedding")
    print("=" * 50)
    
    # Test 1: Import sketch embedding modules
    test1_passed = test_dlrm_sketch_import()
    
    # Test 2: Import DLRM Kaggle script
    test2_passed = test_dlrm_kaggle_import()
    
    # Test 3: Test sketch embedding functionality
    test3_passed = test_sketch_embedding_functionality()
    
    # Test 4: Test DLRM with sketch embedding arguments
    test4_passed = test_dlrm_with_sketch_args()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Sketch embedding import: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"DLRM Kaggle import: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Sketch embedding functionality: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"DLRM sketch arguments: {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 DLRM Kaggle is ready to use with sketch embedding!")
        print("\nExample usage:")
        print("python dlrm_kaggle.py --use_sketch_embedding --batch_size 32 --epochs 1")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 