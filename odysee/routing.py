import numpy as np
import torch
from typing import Optional, Tuple, List, Union, Dict
from odysee_rust import MultiModalRouter as RustRouter
from .cuda_ops import quantum_phase_encoding, flash_attention, to_numpy
from .core.attention import MultiHeadAttention

__all__ = ['MultiModalRouter', 'DynamicRouter']

class DynamicRouter:
    """Advanced routing mechanism for handling multimodal inputs with cross-modal attention."""
    def __init__(self, config):
        self.config = config
        self.router = MultiModalRouter(config.routing_dim, config.num_heads)
        self.cross_attention = MultiHeadAttention(
            hidden_size=config.routing_dim,
            num_heads=config.num_heads,
            use_flash=True
        )
        
    def route(self, input_data: Union[np.ndarray, torch.Tensor], is_image: bool = False,
              cross_modal_data: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Route input data with optional cross-modal attention."""
        try:
            # Convert input to PyTorch tensor if needed
            if isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data)
                
            # Apply quantum phase encoding
            encoded_data = quantum_phase_encoding(input_data)
            
            # Apply cross-modal attention if available
            if cross_modal_data:
                for modality, data in cross_modal_data.items():
                    if data is not None:
                        encoded_data = self.cross_attention(encoded_data, data, data)
            
            if is_image:
                return self._route_image(encoded_data)
            return self._route_text(encoded_data)
            
        except Exception as e:
            raise RuntimeError(f"Routing failed: {str(e)}")
        
    def _route_text(self, text_data: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Route text data through the router."""
        try:
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
            
        except Exception as e:
            raise RuntimeError(f"Text routing failed: {str(e)}")
        
    def _route_image(self, image_data: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Route image data through the router."""
        try:
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
            
        except Exception as e:
            raise RuntimeError(f"Image routing failed: {str(e)}")


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
        if routing_dim <= 0:
            raise ValueError("routing_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
            
        try:
            self.rust_router = RustRouter(routing_dim, num_heads)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Rust router: {e}")
            
        self.routing_dim = routing_dim
        self.num_heads = num_heads
        
    def route_text(
            self, 
            queries: Union[np.ndarray, torch.Tensor],
            batch_size: Optional[int] = None,
            seq_len: Optional[int] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Route text queries to the most relevant experts.
        
        Args:
            queries: Text queries of shape (batch_size * seq_len, routing_dim)
            batch_size: Batch size (optional, inferred if not provided)
            seq_len: Sequence length (optional, inferred if not provided)
            
        Returns:
            Tuple of (routing_weights, routing_indices)
        
        Raises:
            ValueError: If input dimensions are invalid
            RuntimeError: If Rust routing fails
        """
        # Convert torch tensor to numpy if needed
        if isinstance(queries, torch.Tensor):
            queries = queries.detach().cpu().numpy()
            
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
            
        # Validate input dimensions
        total_queries = queries.shape[0]
        if queries.shape[1] != self.routing_dim:
            raise ValueError(f"Expected routing_dim={self.routing_dim}, got {queries.shape[1]}")
            
        # Infer batch_size and seq_len if not provided
        if batch_size is None:
            if seq_len is not None:
                batch_size = total_queries // seq_len
            else:
                batch_size = 1
                seq_len = total_queries
        elif seq_len is None:
            seq_len = total_queries // batch_size
            
        if batch_size * seq_len != total_queries:
            raise ValueError(f"batch_size * seq_len ({batch_size} * {seq_len}) must equal total queries ({total_queries})")
            
        try:
            # Convert to contiguous array and flatten
            queries = np.ascontiguousarray(queries)
            queries_flat = queries.reshape(-1)
            
            # Call Rust router
            weights, indices = self.rust_router.route_text(queries_flat.tolist(), batch_size, seq_len)
            
            # Convert back to numpy arrays
            weights = np.array(weights, dtype=np.float32).reshape(total_queries, self.num_heads)
            indices = np.array(indices, dtype=np.int64).reshape(total_queries, self.num_heads)
            
            return weights, indices
        except Exception as e:
            raise RuntimeError(f"Text routing failed: {e}")
            
    def route_image(
            self,
            queries: Union[np.ndarray, torch.Tensor],
            image_size: Optional[Tuple[int, int]] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Route image queries to the most relevant experts.
        
        Args:
            queries: Image queries of shape (height, width, routing_dim)
            image_size: Image size as (height, width) (optional, inferred if not provided)
            
        Returns:
            Tuple of (routing_weights, routing_indices)
            
        Raises:
            ValueError: If input dimensions are invalid
            RuntimeError: If Rust routing fails
        """
        # Convert torch tensor to numpy if needed
        if isinstance(queries, torch.Tensor):
            queries = queries.detach().cpu().numpy()
            
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
            
        # Handle 4D input (batch, height, width, dim)
        if len(queries.shape) == 4:
            if queries.shape[0] != 1:
                raise ValueError("Currently only supports batch size 1 for image routing")
            queries = queries[0]  # Take first batch
            
        # Validate input dimensions
        if len(queries.shape) != 3:
            raise ValueError(f"Expected 3D input (height, width, routing_dim), got shape {queries.shape}")
            
        if queries.shape[2] != self.routing_dim:
            raise ValueError(f"Expected routing_dim={self.routing_dim}, got {queries.shape[2]}")
            
        # Use provided image_size or infer from input
        if image_size is None:
            image_size = (queries.shape[0], queries.shape[1])
        else:
            if len(image_size) != 2:
                raise ValueError("image_size must be a tuple of (height, width)")
            if image_size[0] * image_size[1] != queries.shape[0] * queries.shape[1]:
                raise ValueError(f"image_size {image_size} does not match input shape {queries.shape[:2]}")
                
        try:
            # Convert to contiguous array and flatten
            queries = np.ascontiguousarray(queries)
            queries_flat = queries.reshape(-1)
            
            # Calculate number of patches (16x16)
            height, width = image_size
            num_patches = ((height + 15) // 16) * ((width + 15) // 16)
            
            # Call Rust router with flattened array and tuple
            weights, indices = self.rust_router.route_image(queries_flat.tolist(), (height, width))
            
            # Convert back to numpy arrays with proper patch dimensions
            weights = np.array(weights, dtype=np.float32).reshape(num_patches, self.num_heads)
            indices = np.array(indices, dtype=np.int64).reshape(num_patches, self.num_heads)
            
            return weights, indices
        except Exception as e:
            raise RuntimeError(f"Image routing failed: {e}")


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
