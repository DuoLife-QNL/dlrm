#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

try:
    from .sketch_embedding_config import SketchEmbeddingBagConfig
    from .sketch_embedding_bag import SKEmbeddingBag
except ImportError:
    from sketch_embedding_config import SketchEmbeddingBagConfig
    from sketch_embedding_bag import SKEmbeddingBag

class SketchEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """
    FX-traceable SketchEmbeddingBagCollection: no Python control flow in forward.
    """
    def __init__(
        self,
        tables: List[SketchEmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        use_cpp: bool = True,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"cafe.{self.__class__.__name__}")
        self._is_weighted = is_weighted
        self._device = device or torch.device("cpu")
        self._embedding_bag_configs = tables
        self._use_cpp = use_cpp

        # Build embedding bags as ModuleList
        self.embedding_bags = nn.ModuleList()
        self._embedding_names: List[str] = []
        self._lengths_per_embedding: List[int] = []
        self._feature_to_bag_and_idx: List[tuple] = []  # For each flat feature, (bag_idx, feature_idx_in_bag)
        bag_feature_names: List[List[str]] = []

        for bag_idx, config in enumerate(tables):
            bag = SKEmbeddingBag(config=config, device=device, use_cpp=use_cpp)
            self.embedding_bags.append(bag)
            bag_feature_names.append(config.feature_names)
            for feature_idx, feature_name in enumerate(config.feature_names):
                self._embedding_names.append(feature_name)
                self._lengths_per_embedding.append(config.embedding_dim)
                self._feature_to_bag_and_idx.append((bag_idx, feature_idx))
        self._bag_feature_names = bag_feature_names

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        feature_dict = features.to_dict()
        pooled_embeddings = []
        # For each feature in flat order, use static mapping to get bag and feature idx
        for i in range(len(self._embedding_names)):
            bag_idx, feature_idx = self._feature_to_bag_and_idx[i]
            feature_name = self._embedding_names[i]
            f = feature_dict[feature_name]
            bag = self.embedding_bags[bag_idx]
            # SKEmbeddingBag always expects single feature per call
            res = bag(
                input=f.values(),
                offsets=f.offsets(),
                per_sample_weights=f.weights() if self._is_weighted else None,
            ).float()
            pooled_embeddings.append(res)
        combined = torch.cat(pooled_embeddings, dim=1)
        return KeyedTensor(
            keys=self._embedding_names,
            values=combined,
            length_per_key=self._lengths_per_embedding,
        )

    def is_weighted(self) -> bool:
        return self._is_weighted

    def embedding_bag_configs(self) -> List[SketchEmbeddingBagConfig]:
        return self._embedding_bag_configs

    @property
    def device(self) -> torch.device:
        return self._device

    def update_all_sketches_with_gradients(self):
        for embedding_bag in self.embedding_bags:
            embedding_bag.update_sketch_with_gradients()

    def get_sketch_stats(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for i, embedding_bag in enumerate(self.embedding_bags):
            stats[self._embedding_names[i]] = embedding_bag.get_sketch_info()
        return stats

    def reset_parameters(self) -> None:
        for embedding_bag in self.embedding_bags:
            embedding_bag.reset_parameters()

    def extra_repr(self) -> str:
        lines = [
            f"tables={len(self._embedding_bag_configs)}",
            f"is_weighted={self._is_weighted}",
        ]
        total_hot_size = sum(config.hot_size for config in self._embedding_bag_configs)
        total_hash_size = sum(config.hash_size for config in self._embedding_bag_configs)
        lines.extend([
            f"total_hot_size={total_hot_size}",
            f"total_hash_size={total_hash_size}",
        ])
        return ", ".join(lines)


def create_sketch_embedding_bag_collection(
    num_embeddings_per_feature: List[int],
    embedding_dim: int,
    feature_names: List[str],
    compress_rate: float = 0.1,
    hash_rate: float = 0.3,
    sketch_threshold: int = 1,
    sketch_alpha: float = 1.0000005,
    device: Optional[torch.device] = None,
) -> SketchEmbeddingBagCollection:
    """
    Convenience function to create a SketchEmbeddingBagCollection.
    
    Args:
        num_embeddings_per_feature: Number of embeddings for each feature
        embedding_dim: Embedding dimension
        feature_names: List of feature names
        compress_rate: Compression rate for hot table sizing
        hash_rate: Hash rate for hash table sizing
        sketch_threshold: Sketch threshold parameter
        sketch_alpha: Sketch alpha parameter
        device: Device to place embeddings on
        
    Returns:
        SketchEmbeddingBagCollection instance
    """
    tables = []
    
    for i, (num_emb, feature_name) in enumerate(zip(num_embeddings_per_feature, feature_names)):
        hot_size = int(num_emb * compress_rate)
        hash_size = int(num_emb * hash_rate)
        
        config = SketchEmbeddingBagConfig(
            name=f"table_{i}",
            embedding_dim=embedding_dim,
            num_embeddings=num_emb,
            feature_names=[feature_name],
            hot_size=hot_size,
            hash_size=hash_size,
            compress_rate=compress_rate,
            sketch_threshold=sketch_threshold,
            sketch_alpha=sketch_alpha,
            hash_rate=hash_rate,
        )
        tables.append(config)
        
    return SketchEmbeddingBagCollection(
        tables=tables,
        device=device,
    ) 