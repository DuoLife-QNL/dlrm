#!/usr/bin/env python3

import ctypes
import numpy as np
import torch
import torch.fx
from typing import Optional, Tuple
import os
import warnings


class SketchOperations:
    """
    Sketch-based operations for managing hot/cold feature promotion.
    
    This class provides C++ accelerated sketch operations for efficient
    feature frequency tracking and hot/cold table management.
    """
    
    def __init__(self, hot_size=1000, hash_size=5000, sketch_threshold=100, 
                 adjust_threshold=50, sketch_alpha=1.0, use_cpp=True):
        """
        Initialize sketch operations.
        
        Args:
            hot_size: Size of hot embedding table
            hash_size: Size of hash embedding table  
            sketch_threshold: Threshold for hot feature promotion
            adjust_threshold: Threshold for sketch adjustment
            sketch_alpha: Alpha parameter for sketch decay
            use_cpp: Whether to use C++ acceleration
        """
        self.hot_size = hot_size
        self.hash_size = hash_size
        self.sketch_threshold = sketch_threshold
        self.adjust_threshold = adjust_threshold
        self.sketch_alpha = sketch_alpha
        self._use_cpp = use_cpp
        self._initialized = False
        self._lib = None
        
        if use_cpp:
            self._load_cpp_library()
        else:
            raise RuntimeError("C++ acceleration is required. Set use_cpp=True")
            
        # Clean up any existing C++ state
        if self._lib is not None:
            try:
                # Only clean up once when library is first loaded
                self._lib.cleanup()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize C++ sketch: {e}")
        
        self._initialize_sketch()
        
    def _is_fx_tracing(self) -> bool:
        """Check if we are currently in FX tracing mode."""
        try:
            return torch.fx._symbolic_trace.is_fx_tracing()
        except:
            # Fallback check for older PyTorch versions
            return hasattr(torch.fx, '_symbolic_trace') and torch.fx._symbolic_trace.is_fx_tracing()
        
    def _load_cpp_library(self):
        """Load the C++ sketch library."""
        try:
            # Try to load compiled library
            lib_path = os.path.join(os.path.dirname(__file__), "libsketch.so")
            if os.path.exists(lib_path):
                self._lib = ctypes.CDLL(lib_path)
                self._setup_function_signatures()
                self._initialize_sketch()
            else:
                raise RuntimeError(
                    f"C++ sketch library not found at {lib_path}. "
                    "Please compile the C++ library first."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load C++ sketch library: {e}")
            
    def _setup_function_signatures(self):
        """Setup C++ function signatures."""
        if self._lib is None:
            return
            
        # Setup init function
        self._lib.init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self._lib.init.restype = None
        
        # Setup cleanup function
        self._lib.cleanup.argtypes = []
        self._lib.cleanup.restype = None
        
        # Setup batch_query function
        self._lib.batch_query.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
        self._lib.batch_query.restype = ctypes.POINTER(ctypes.c_int)
        
        # Setup batch_insert function
        self._lib.batch_insert.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
        self._lib.batch_insert.restype = ctypes.POINTER(ctypes.c_int)
        
        # Setup batch_insert_val function
        self._lib.batch_insert_val.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), 
            ctypes.POINTER(ctypes.c_float), 
            ctypes.c_int
        ]
        self._lib.batch_insert_val.restype = ctypes.POINTER(ctypes.c_int)
        
    def _initialize_sketch(self):
        """Initialize the C++ sketch data structure."""
        if self._lib is None:
            raise RuntimeError("C++ library not loaded")
            
        # Clean up any existing state first
        self._lib.cleanup()
        
        # Ensure adjust_threshold is int type
        adjust_threshold_int = int(self.adjust_threshold)
        
        self._lib.init(
            self.hot_size,
            self.sketch_threshold, 
            adjust_threshold_int,  # Convert to int
            self.sketch_alpha
        )
        self._initialized = True
        
    def batch_query(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Query sketch for given indices.
        
        Args:
            indices: Tensor of indices to query
            
        Returns:
            Tensor with query results (negative values indicate hot features)
        """
        # Skip sketch operations during FX tracing
        if self._is_fx_tracing():
            raise RuntimeError("Sketch operations not supported during FX tracing")
            
        if not self._initialized:
            raise RuntimeError("Sketch not initialized")
            
        # Convert to numpy and ensure correct dtype
        indices_np = indices.detach().cpu().numpy().astype(np.uint32)
        N = len(indices_np)
        
        # Get data pointer
        indices_ptr = indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        # Call C++ function
        result_ptr = self._lib.batch_query(indices_ptr, N)
        
        # Convert result back to torch tensor
        result_np = np.ctypeslib.as_array(result_ptr, shape=(N,))
        result = torch.from_numpy(result_np.copy()).to(indices.device)
        
        return result
        
    def batch_insert(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Insert features into sketch.
        
        Args:
            indices: Tensor of indices to insert
            
        Returns:
            Tensor indicating which features became hot (1) or stayed cold (0)
        """
        # Skip sketch operations during FX tracing
        if self._is_fx_tracing():
            raise RuntimeError("Sketch operations not supported during FX tracing")
            
        if not self._initialized:
            raise RuntimeError("Sketch not initialized")
            
        # Convert to numpy and ensure correct dtype
        indices_np = indices.detach().cpu().numpy().astype(np.uint32)
        N = len(indices_np)
        
        # Get data pointer
        indices_ptr = indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        # Call C++ function
        result_ptr = self._lib.batch_insert(indices_ptr, N)
        
        # Convert result back to torch tensor
        result_np = np.ctypeslib.as_array(result_ptr, shape=(N,))
        result = torch.from_numpy(result_np.copy()).to(indices.device)
        
        return result
        
    def batch_insert_with_gradients(
        self, 
        indices: torch.Tensor, 
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Insert features with gradient information.
        
        Args:
            indices: Tensor of indices to insert
            gradients: Gradient norms for the features
            
        Returns:
            Tensor indicating which features became hot
        """
        # Skip sketch operations during FX tracing
        if self._is_fx_tracing():
            raise RuntimeError("Sketch operations not supported during FX tracing")
            
        if not self._initialized:
            raise RuntimeError("Sketch not initialized")
            
        # Convert to numpy
        indices_np = indices.detach().cpu().numpy().astype(np.uint32)
        gradients_np = gradients.detach().cpu().numpy().astype(np.float32)
        N = len(indices_np)
        
        # Get data pointers
        indices_ptr = indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        gradients_ptr = gradients_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call C++ function
        result_ptr = self._lib.batch_insert_val(indices_ptr, gradients_ptr, N)
        
        # Convert result back to torch tensor
        result_np = np.ctypeslib.as_array(result_ptr, shape=(N,))
        result = torch.from_numpy(result_np.copy()).to(indices.device)
        
        return result
        
    def is_initialized(self) -> bool:
        """Check if sketch is properly initialized."""
        return self._initialized
        
    def has_cpp_acceleration(self) -> bool:
        """Check if C++ acceleration is available."""
        return self._lib is not None 