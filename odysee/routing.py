import numpy as np
import torch
from typing import Optional, Tuple, List, Union
from odysee_rust import MultiModalRouter as RustRouter
from .cuda_ops import quantum_phase_encoding, flash_attention, to_numpy

__all__ = ['MultiModalRouter', 'DynamicRouter']

class DynamicRouter:
    """Advanced routing mechanism for handling both text and images with 4M context windows."""
    def __init__(self, config):
        self.config = config
        self.router = MultiModalRouter(config.routing_dim, config.num_heads)
        
    def route(self, input_data: Union[np.ndarray, torch.Tensor], is_image: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Route input data to appropriate experts."""
        # Convert input to PyTorch tensor if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
            
        # Apply quantum phase encoding
        encoded_data = quantum_phase_encoding(input_data)
        
        if is_image:
            return self._route_image(encoded_data)
        return self._route_text(encoded_data)
        
    def _route_text(self, text_data: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Route text data through the router."""
        batch_size = text_data.shape[0]
        seq_len = text_data.shape[1]
        
        # Generate attention patterns using flash attention
        query = text_data
        key = value = text_data
        
        attention_output = flash_attention(
            query, key, value,
            num_heads=self.config.num_heads
        )
        
        # Convert to numpy for Rust router
        flat_data = to_numpy(attention_output).reshape(-1, self.config.routing_dim)
        
        weights, indices = self.router.route_text(
            flat_data.astype(np.float32),
            batch_size,
            seq_len
        )
        
        return weights, indices
        
    def _route_image(self, image_data: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Route image data through the router."""
        height, width = image_data.shape[:2]
        
        # Generate attention patterns using flash attention
        query = image_data.view(1, height * width, -1)
        key = value = query
        
        attention_output = flash_attention(
            query, key, value,
            num_heads=self.config.num_heads
        )
        
        # Convert to numpy for Rust router
        flat_data = to_numpy(attention_output).reshape(-1, self.config.routing_dim)
        
        weights, indices = self.router.route_text(
            flat_data.astype(np.float32),
            batch_size=1,
            seq_len=height * width
        )
        
        return weights, indices


class MultiModalRouter:
    """
    Python wrapper for the Rust MultiModalRouter.
    Provides a high-level interface for routing multimodal inputs.
    """
    def __init__(self, routing_dim: int, num_heads: int = 4):
        """Initialize the multi-modal router.
        
        Args:
            routing_dim: Dimension of the routing embeddings
            num_heads: Number of routing heads (default: 4)
        """
        self.rust_router = RustRouter(routing_dim, num_heads)
        self.routing_dim = routing_dim
        self.num_heads = num_heads
        
    def route_text(
        self, 
        queries: Union[np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route text queries to the most relevant experts.
        
        Args:
            queries: Text queries of shape (batch_size * seq_len, routing_dim)
            batch_size: Batch size (optional, inferred if not provided)
            seq_len: Sequence length (optional, inferred if not provided)
            
        Returns:
            Tuple of (routing_weights, routing_indices)
        """
        # Convert to numpy if needed
        queries = to_numpy(queries)
            
        # Ensure float32 type and correct shape
        queries = queries.astype(np.float32)
        
        # Handle different input shapes
        if len(queries.shape) == 3:  # (batch, seq, dim)
            batch_size = queries.shape[0]
            seq_len = queries.shape[1]
            queries = queries.reshape(-1, queries.shape[-1])
        elif len(queries.shape) == 2:  # (batch*seq, dim)
            if batch_size is None:
                batch_size = 1
            if seq_len is None:
                seq_len = queries.shape[0] // batch_size
        else:
            raise ValueError(f"Invalid query shape: {queries.shape}")
            
        # Ensure we have valid dimensions
        if queries.shape[0] != batch_size * seq_len:
            raise ValueError(f"Query shape {queries.shape} does not match batch_size={batch_size} and seq_len={seq_len}")
            
        # Flatten for Rust router
        queries_flat = queries.reshape(-1).tolist()
        
        # Call Rust router
        weights, indices = self.rust_router.route_text(queries_flat, batch_size, seq_len)
        
        # Convert back to numpy arrays with proper shape
        weights = np.array(weights).reshape(batch_size * seq_len, -1)
        indices = np.array(indices).reshape(batch_size * seq_len, -1)
        
        return weights, indices
        
    def route_image(
        self,
        queries: Union[np.ndarray, torch.Tensor],
        image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route image queries to the most relevant experts.
        
        Args:
            queries: Image queries of shape (height, width, routing_dim)
            image_size: Image size as (height, width) (optional, inferred if not provided)
            
        Returns:
            Tuple of (routing_weights, routing_indices)
        """
        # Convert to numpy if needed
        queries = to_numpy(queries)
            
        # Ensure float32 type and correct shape
        queries = queries.astype(np.float32)
        
        # Handle different input shapes
        if len(queries.shape) == 4:  # (batch, height, width, dim)
            if image_size is None:
                image_size = (queries.shape[1], queries.shape[2])
            queries = queries.reshape(-1, queries.shape[-1])
        elif len(queries.shape) == 3:  # (height, width, dim)
            if image_size is None:
                image_size = (queries.shape[0], queries.shape[1])
            queries = queries.reshape(-1, queries.shape[-1])
        elif len(queries.shape) == 2:  # (height*width, dim)
            if image_size is None:
                # Assume square image
                side_len = int(np.sqrt(queries.shape[0]))
                image_size = (side_len, side_len)
        else:
            raise ValueError(f"Invalid query shape: {queries.shape}")
            
        # Ensure we have valid dimensions
        if queries.shape[0] != image_size[0] * image_size[1]:
            raise ValueError(f"Query shape {queries.shape} does not match image_size={image_size}")
            
        # Flatten for Rust router
        queries_flat = queries.reshape(-1).tolist()
        
        # Call Rust router
        weights, indices = self.rust_router.route_image(queries_flat, image_size)
        
        # Convert back to numpy arrays with proper shape
        weights = np.array(weights).reshape(image_size[0] * image_size[1], -1)
        indices = np.array(indices).reshape(image_size[0] * image_size[1], -1)
        
        return weights, indices


from dataclasses import dataclass

@dataclass
class RoutingConfig:
    """Configuration for the multi-modal router"""
    routing_dim: int = 1024
    num_heads: int = 8
    chunk_size: int = 4096
    max_context_length: int = 4_000_000
    image_patch_size: int = 16
    use_flash_attention: bool = True
    dropout: float = 0.1
