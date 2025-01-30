import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class DynamicRouter(nn.Module):
    """
    Dynamic routing mechanism that adaptively routes information through the network.
    Features:
    1. Content-based routing with Metal acceleration
    2. Memory-efficient sparse routing
    3. Hierarchical information processing with gradient checkpointing
    """
    
    def __init__(
        self,
        config,
        use_checkpointing: bool = True
    ):
        super().__init__()
        self.num_routes = config.routing_heads
        self.route_dim = config.hidden_size // config.routing_heads
        self.scale = math.sqrt(self.route_dim)
        
        # Routing components
        self.route_queries = nn.Linear(config.hidden_size, config.hidden_size)
        self.route_keys = nn.Linear(config.hidden_size, config.hidden_size)
        self.route_values = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Learnable temperature parameter for sharpening routing decisions
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Route aggregation with layer normalization for better stability
        self.route_combine = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Gating mechanism for adaptive information flow
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid()
        )
        
        self.use_checkpointing = use_checkpointing
        
    def _compute_routing_weights(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        top_k: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse routing weights using efficient Metal operations.
        Args:
            queries: (batch_size, seq_len, hidden_size)
            keys: (batch_size, num_routes, hidden_size)
            top_k: number of top routes to select
        Returns:
            routing_weights: (batch_size, seq_len, num_routes)
            routing_indices: (batch_size, seq_len, top_k)
        """
        # Move tensors to Metal device if available
        device = torch.device("mps") if torch.backends.mps.is_available() else queries.device
        queries = queries.to(device)
        keys = keys.to(device)
        
        # Compute routing logits with scaled dot-product
        routing_logits = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        
        # Get top-k routes
        routing_logits_topk, indices = torch.topk(routing_logits, top_k, dim=-1)
        
        # Apply temperature scaling and compute sparse softmax
        routing_weights = torch.zeros_like(routing_logits).scatter_(
            -1, indices,
            F.softmax(routing_logits_topk / self.temperature, dim=-1)
        )
        
        return routing_weights, indices
    
    def _route_information(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        chunk_size: int = 512
    ) -> torch.Tensor:
        """
        Route information through the network in memory-efficient chunks.
        Args:
            x: (batch_size, seq_len, hidden_size)
            memory: optional external memory to route through
            chunk_size: size of sequence chunks for memory efficiency
        Returns:
            routed_info: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Process in chunks for memory efficiency
        chunks = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk = x[:, chunk_start:chunk_end]
            
            # Compute queries and keys
            queries = self.route_queries(chunk)
            keys = self.route_keys(memory) if memory is not None else self.route_keys(x)
            values = self.route_values(memory) if memory is not None else self.route_values(x)
            
            # Get routing weights
            routing_weights, _ = self._compute_routing_weights(queries, keys)
            
            # Route information
            routed_chunk = torch.matmul(routing_weights, values)
            chunks.append(routed_chunk)
        
        # Combine chunks
        routed_info = torch.cat(chunks, dim=1)
        
        return routed_info
    
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with gradient checkpointing for memory efficiency.
        Args:
            x: (batch_size, seq_len, hidden_size)
            memory: optional external memory
        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        if self.use_checkpointing and self.training:
            routed_info = torch.utils.checkpoint.checkpoint(
                self._route_information,
                x, memory
            )
        else:
            routed_info = self._route_information(x, memory)
        
        # Combine routed information with input
        combined_info = self.route_combine(routed_info)
        combined_info = self.layer_norm(combined_info)
        combined_info = self.dropout(combined_info)
        
        # Compute adaptive gates
        gate_input = torch.cat([x, combined_info], dim=-1)
        gates = self.gate(gate_input)
        
        # Gated combination of input and routed information
        output = gates * combined_info + (1 - gates) * x
        
        return output


from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import math
from dataclasses import dataclass
from .rust.routing import MultiModalRouter

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

class MultiModalDynamicRouter:
    """
    Advanced routing mechanism for handling both text and images with 4M context windows.
    Uses Rust backend for efficient processing and Metal acceleration where available.
    
    Features:
    1. Efficient 4M token context handling through chunked processing
    2. Multi-modal routing supporting both text and images
    3. Memory-efficient sparse routing with flash attention
    4. Hierarchical information processing
    """
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.router = MultiModalRouter(config.routing_dim, config.num_heads)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize routing embeddings"""
        # Text position embeddings for relative positions
        max_len = min(16384, self.config.max_context_length)  # Local context size
        self.text_pos_embedding = np.random.normal(
            0, 0.02, 
            (max_len, self.config.routing_dim)
        ).astype(np.float32)
        
        # Image position embeddings for 2D positions
        max_h = max_w = 4096 // self.config.image_patch_size
        self.image_pos_embedding = np.random.normal(
            0, 0.02,
            (max_h, max_w, self.config.routing_dim)
        ).astype(np.float32)
    
    def _process_text_chunk(
        self,
        chunk: np.ndarray,
        chunk_start: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a chunk of text with relative positions"""
        # Add relative position information
        chunk_len = len(chunk)
        pos_emb = self.text_pos_embedding[chunk_start:chunk_start + chunk_len]
        chunk_with_pos = chunk + pos_emb
        
        # Get routing weights and indices from Rust
        weights, indices = self.router.route_text(
            chunk_with_pos.ravel().tolist(),
            1,  # batch size
            chunk_len
        )
        
        return np.array(weights), np.array(indices)
    
    def _process_image_patch(
        self,
        patch: np.ndarray,
        h_idx: int,
        w_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process an image patch with 2D position information"""
        # Add 2D position information
        pos_emb = self.image_pos_embedding[h_idx, w_idx]
        patch_with_pos = patch + pos_emb
        
        # Get routing weights and indices from Rust
        weights, indices = self.router.route_image(
            patch_with_pos.ravel().tolist(),
            patch.shape[:2]
        )
        
        return np.array(weights), np.array(indices)
    
    def route(
        self,
        inputs: Union[np.ndarray, Image.Image],
        memory: Optional[np.ndarray] = None,
        is_image: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Route information through the network.
        
        Args:
            inputs: Text embeddings (batch_size, seq_len, dim) or PIL Image
            memory: Optional external memory to route through
            is_image: Whether the input is an image
            
        Returns:
            routed_info: Routed information with same shape as input
            routing_stats: Dictionary with routing statistics
        """
        if is_image:
            # Convert PIL image to numpy array if needed
            if isinstance(inputs, Image.Image):
                inputs = np.array(inputs)
            
            # Update image embeddings in router
            if memory is not None:
                self.router.update_embeddings(
                    None,
                    memory.ravel().tolist(),
                    None,
                    memory.shape
                )
            
            # Process image in patches
            h, w = inputs.shape[:2]
            patch_size = self.config.image_patch_size
            num_patches_h = (h + patch_size - 1) // patch_size
            num_patches_w = (w + patch_size - 1) // patch_size
            
            routed_patches = []
            routing_weights = []
            routing_indices = []
            
            for h_idx in range(num_patches_h):
                patch_row = []
                h_start = h_idx * patch_size
                h_end = min(h_start + patch_size, h)
                
                for w_idx in range(num_patches_w):
                    w_start = w_idx * patch_size
                    w_end = min(w_start + patch_size, w)
                    
                    patch = inputs[h_start:h_end, w_start:w_end]
                    weights, indices = self._process_image_patch(patch, h_idx, w_idx)
                    
                    # Route patch information
                    if memory is not None:
                        routed_patch = np.sum(
                            memory[indices] * weights[:, None, None, None],
                            axis=0
                        )
                    else:
                        routed_patch = patch
                    
                    patch_row.append(routed_patch)
                    routing_weights.append(weights)
                    routing_indices.append(indices)
                
                routed_patches.append(patch_row)
            
            # Combine patches
            routed_info = np.block(routed_patches)[:h, :w]
            
        else:
            # Update text embeddings in router
            if memory is not None:
                self.router.update_embeddings(
                    memory.ravel().tolist(),
                    None,
                    memory.shape,
                    None
                )
            
            # Process text in chunks
            seq_len = inputs.shape[1]
            chunk_size = self.config.chunk_size
            
            routed_chunks = []
            routing_weights = []
            routing_indices = []
            
            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk = inputs[:, chunk_start:chunk_end]
                
                weights, indices = self._process_text_chunk(chunk, chunk_start)
                
                # Route chunk information
                if memory is not None:
                    routed_chunk = np.sum(
                        memory[indices] * weights[:, None],
                        axis=0
                    )
                else:
                    routed_chunk = chunk
                
                routed_chunks.append(routed_chunk)
                routing_weights.append(weights)
                routing_indices.append(indices)
            
            # Combine chunks
            routed_info = np.concatenate(routed_chunks, axis=0)
            routed_info = routed_info.reshape(inputs.shape)
        
        # Compute routing statistics
        routing_stats = {
            "sparsity": np.mean([np.count_nonzero(w) / len(w) for w in routing_weights]),
            "entropy": np.mean([-(w * np.log(w + 1e-10)).sum() for w in routing_weights]),
            "max_weight": np.max([np.max(w) for w in routing_weights]),
            "num_routes": len(set().union(*[set(i.tolist()) for i in routing_indices]))
        }
        
        return routed_info, routing_stats
