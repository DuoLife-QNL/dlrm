#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
from torch.nn.parameter import Parameter
from typing import Optional, Dict, Any
import numpy as np

try:
    from .sketch_ops import SketchOperations
    from .sketch_embedding_config import SketchEmbeddingBagConfig
except ImportError:
    from sketch_ops import SketchOperations
    from sketch_embedding_config import SketchEmbeddingBagConfig


class SKEmbeddingBag(nn.Module):
    """
    TorchRec-compatible Sketch-based EmbeddingBag implementation.
    
    This module implements the CAFE sketch-based embedding compression technique
    within TorchRec's framework. It maintains separate embedding tables for hot
    and cold features, dynamically promoting features based on access frequency.
    """
    
    def __init__(
        self,
        config: SketchEmbeddingBagConfig,
        device: Optional[torch.device] = None,
        use_cpp: bool = True,
    ):
        """
        Initialize SKEmbeddingBag.
        
        Args:
            config: SketchEmbeddingBagConfig containing all necessary parameters
            device: Device to place the embedding tables on
            use_cpp: Whether to use C++ acceleration for sketch operations
        """
        super(SKEmbeddingBag, self).__init__()
        
        self.config = config
        self.device = device or torch.device("cpu")
        self.use_cpp = use_cpp
        
        # Embedding parameters
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embedding_dim
        self.hot_size = config.hot_size
        self.hash_size = config.hash_size
        
        # Initialize sketch operations
        self.sketch_ops = SketchOperations(
            hot_size=config.hot_size,
            sketch_threshold=config.sketch_threshold,
            adjust_threshold=config.adjust_threshold,
            sketch_alpha=config.sketch_alpha,
            use_cpp=use_cpp,
        )
        
        # Create embedding tables
        self._create_embedding_tables()
        
        # State tracking
        self.register_buffer('_training_step', torch.tensor(0, dtype=torch.long))
        self._last_indices = None
        
    def _is_fx_tracing(self, input_tensor: torch.Tensor) -> bool:
        """Check if we are currently in FX tracing mode by examining input tensor."""
        return 'Proxy' in str(type(input_tensor))
        
    def _create_embedding_tables(self):
        """Create hot and hash embedding tables."""
        # Hot embedding table for frequently accessed features
        self.weight_hot = Parameter(
            torch.empty(self.hot_size, self.embedding_dim, device=self.device)
        )
        
        # Hash embedding table for cold features  
        self.weight_hash = Parameter(
            torch.empty(self.hash_size, self.embedding_dim, device=self.device)
        )
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize embedding weights."""
        # Use the same initialization as standard EmbeddingBag
        init_range = np.sqrt(1 / self.num_embeddings)
        
        nn.init.uniform_(self.weight_hot, -init_range, init_range)
        nn.init.uniform_(self.weight_hash, -init_range, init_range)
        
    def forward(
        self,
        input: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        per_sample_weights: Optional[torch.Tensor] = None,
        test: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of SKEmbeddingBag.
        
        Args:
            input: Input indices tensor
            offsets: Offsets tensor for variable-length sequences
            per_sample_weights: Per-sample weights for weighted pooling
            test: Whether in testing mode (no sketch updates)
            
        Returns:
            Pooled embeddings tensor
        """
        if offsets is None:
            offsets = torch.arange(
                0, input.size(0) + 1, dtype=torch.long, device=input.device
            )
            
        # Store indices for potential gradient-based updates
        self._last_indices = input
        
        # Skip sketch operations during FX tracing
        if self._is_fx_tracing(input):
            return self._fx_trace_forward(input, offsets, per_sample_weights)
        
        if not test and self.training:
            # Training mode: update sketch and handle hot/cold promotion
            return self._forward_training(input, offsets, per_sample_weights)
        else:
            # Inference mode: only query sketch
            return self._forward_inference(input, offsets, per_sample_weights)
            
    def _fx_trace_forward(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass during FX tracing - skip sketch operations."""
        # During FX tracing, just use hash table lookup as fallback
        return F.embedding_bag(
            input % self.hash_size,
            self.weight_hash,
            offsets,
            sparse=False,
            per_sample_weights=per_sample_weights,
            include_last_offset=True,
        )
            
    def _forward_training(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass during training with sketch updates."""
        
        # Insert features into sketch to update frequency counts
        promotion_mask = self.sketch_ops.batch_insert(input)
        
        # Query sketch to get hot/cold assignments
        query_result = self.sketch_ops.batch_query(input)
        is_hot = query_result < 0
        hot_indices = torch.abs(query_result)
        
        # Handle newly promoted features
        if promotion_mask.sum() > 0:
            self._handle_promotion(input, promotion_mask, hot_indices)
            
        # Perform embedding lookup
        return self._embedding_lookup(input, hot_indices, is_hot, offsets, per_sample_weights)
        
    def _forward_inference(
        self,
        input: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass during inference (no sketch updates)."""
        
        # Query sketch to get hot/cold assignments
        query_result = self.sketch_ops.batch_query(input)
        is_hot = query_result < 0
        hot_indices = torch.abs(query_result)
        
        # Perform embedding lookup
        return self._embedding_lookup(input, hot_indices, is_hot, offsets, per_sample_weights)
        
    def _embedding_lookup(
        self,
        input: torch.Tensor,
        hot_indices: torch.Tensor,
        is_hot: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform the actual embedding lookup and pooling."""
        
        # Hot embedding lookup
        hot_embed = F.embedding_bag(
            hot_indices % self.hot_size,
            self.weight_hot,
            offsets,
            sparse=False,
            per_sample_weights=per_sample_weights,
            include_last_offset=True,
        )
        
        # Hash embedding lookup  
        hash_embed = F.embedding_bag(
            hot_indices % self.hash_size,
            self.weight_hash,
            offsets,
            sparse=False,
            per_sample_weights=per_sample_weights,
            include_last_offset=True,
        )
        
        # Combine hot and hash embeddings based on is_hot mask
        # Reshape is_hot to match embedding dimensions
        if len(is_hot.shape) == 1 and len(hot_embed.shape) == 2:
            batch_size = len(offsets) - 1
            if is_hot.shape[0] != batch_size:
                # fallback: treat all as cold, shape [batch, embedding_dim]
                is_hot = torch.zeros(batch_size, hot_embed.size(1), dtype=torch.bool, device=hot_embed.device)
            else:
                bag_is_hot = torch.zeros(batch_size, dtype=torch.bool, device=is_hot.device)
                for i in range(batch_size):
                    start_idx = offsets[i]
                    end_idx = offsets[i + 1]
                    bag_is_hot[i] = is_hot[start_idx:end_idx].any()
                is_hot = bag_is_hot
        if is_hot.dim() == 1:
            is_hot = is_hot.unsqueeze(1).expand(-1, hot_embed.size(1))
        assert hot_embed.shape == hash_embed.shape, f"hot_embed {hot_embed.shape} != hash_embed {hash_embed.shape}"
        assert is_hot.shape == hot_embed.shape, f"is_hot {is_hot.shape} != hot_embed {is_hot.shape}"
        result = torch.where(is_hot, hot_embed, hash_embed)
        return result
        
    def _handle_promotion(
        self,
        input: torch.Tensor,
        promotion_mask: torch.Tensor,
        hot_indices: torch.Tensor,
    ):
        """Handle promotion of cold features to hot table."""
        promoted_features = input[promotion_mask.bool()]
        promoted_hot_indices = hot_indices[promotion_mask.bool()]
        
        if len(promoted_features) > 0:
            with torch.no_grad():
                # Copy weights from hash table to hot table for promoted features
                for orig_idx, hot_idx in zip(promoted_features, promoted_hot_indices):
                    hash_idx = orig_idx % self.hash_size
                    hot_slot = hot_idx % self.hot_size
                    self.weight_hot[hot_slot].copy_(self.weight_hash[hash_idx])
                    
    def update_sketch_with_gradients(self):
        """Update sketch using gradient information."""
        if not self.training or self._last_indices is None:
            return
            
        # Calculate gradient norms for the last batch
        if hasattr(self, 'weight_hot') and self.weight_hot.grad is not None:
            # This is a simplified version - in practice you'd need to carefully
            # track which gradients correspond to which original indices
            grad_norms = torch.norm(self.weight_hot.grad, dim=1, p=2)
            
            # Use gradient information to update sketch
            if len(grad_norms) == len(self._last_indices):
                self.sketch_ops.batch_insert_with_gradients(
                    self._last_indices, grad_norms[:len(self._last_indices)]
                )
                
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f'num_embeddings={self.num_embeddings}, '
            f'embedding_dim={self.embedding_dim}, '
            f'hot_size={self.hot_size}, '
            f'hash_size={self.hash_size}'
        )
        
    def get_sketch_info(self) -> Dict[str, Any]:
        """Get information about the sketch state."""
        return {
            'hot_size': self.hot_size,
            'hash_size': self.hash_size,
            'sketch_initialized': self.sketch_ops.is_initialized(),
            'has_cpp_acceleration': self.sketch_ops.has_cpp_acceleration(),
        } 