#!/usr/bin/env python3
# Copyright (c) CAFE Project Contributors.
# Licensed under the Apache License, Version 2.0.

"""
CAFE: TorchRec integration for sketch-based embedding compression.

This module provides TorchRec-compatible implementations of sketch-based 
embedding compression techniques for large-scale recommendation systems.
"""

from .sketch_embedding_config import SketchEmbeddingBagConfig
from .sketch_embedding_bag import SKEmbeddingBag
from .sketch_embedding_collection import SketchEmbeddingBagCollection
from .sketch_ops import SketchOperations

__all__ = [
    "SketchEmbeddingBagConfig",
    "SKEmbeddingBag", 
    "SketchEmbeddingBagCollection",
    "SketchOperations",
] 