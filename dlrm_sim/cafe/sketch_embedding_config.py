#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, List, Callable
import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig, DataType, PoolingType


@dataclass
class SketchEmbeddingBagConfig(EmbeddingBagConfig):
    """
    SketchEmbeddingBagConfig extends EmbeddingBagConfig with sketch-specific parameters.
    
    This configuration supports the CAFE sketch-based embedding compression technique,
    which dynamically manages hot and cold features using a count sketch data structure.
    
    Args:
        hot_size (int): Size of the hot embedding table for frequently accessed features.
        hash_size (int): Size of the hash embedding table for cold features.
        compress_rate (float): Compression rate for determining hot/cold threshold.
        sketch_threshold (int): Threshold for sketch data structure.
        sketch_alpha (float): Alpha parameter for sketch decay.
        hash_rate (float): Rate for hash table sizing.
        adjust_threshold (int): Adjustment threshold for sketch reset.
    """
    
    # Sketch-specific parameters
    hot_size: int = 10000
    hash_size: int = 50000
    compress_rate: float = 0.1
    sketch_threshold: int = 1
    sketch_alpha: float = 1.0000005
    hash_rate: float = 0.3
    adjust_threshold: int = 1
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # Validate sketch parameters
        if self.hot_size <= 0:
            raise ValueError("hot_size must be positive")
        if self.hash_size <= 0:
            raise ValueError("hash_size must be positive")
        if not 0 < self.compress_rate <= 1:
            raise ValueError("compress_rate must be between 0 and 1")
        if self.sketch_threshold < 1:
            raise ValueError("sketch_threshold must be >= 1")
        if self.sketch_alpha <= 1.0:
            raise ValueError("sketch_alpha must be > 1.0")
            
    def get_total_embeddings(self) -> int:
        """Get the total number of embeddings (hot + cold)."""
        return self.hot_size + self.hash_size
        
    def get_sketch_config(self) -> dict:
        """Get sketch-specific configuration as a dictionary."""
        return {
            "hot_size": self.hot_size,
            "hash_size": self.hash_size,
            "compress_rate": self.compress_rate,
            "sketch_threshold": self.sketch_threshold,
            "sketch_alpha": self.sketch_alpha,
            "hash_rate": self.hash_rate,
            "adjust_threshold": self.adjust_threshold,
        } 