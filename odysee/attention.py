import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
from typing import Optional, Tuple

class MultiScaleAttention(nn.Module):
    """Multi-scale attention module that processes input at different granularities"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        window_sizes: Optional[Tuple[int, ...]] = None,
        use_checkpointing: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_sizes = window_sizes or (8, 16, 32, 64)
        self.use_checkpointing = use_checkpointing
        
        # Create attention modules for each scale
        self.attention_layers = nn.ModuleList([
            MetalOptimizedAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_length=size,
                use_checkpointing=use_checkpointing
            )
            for size in self.window_sizes
        ])
        
        # Output projection
        self.output_proj = nn.Linear(dim * len(self.window_sizes), dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        outputs = []
        
        # Process at each scale
        for i, attn in enumerate(self.attention_layers):
            # Pad sequence length to window size if needed
            window_size = self.window_sizes[i]
            pad_len = (window_size - L % window_size) % window_size
            if pad_len > 0:
                x_pad = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
            else:
                x_pad = x
            
            # Apply attention
            output, _ = attn(x_pad)
            
            # Remove padding
            if pad_len > 0:
                output = output[:, :L]
            
            outputs.append(output)
        
        # Combine outputs from different scales
        multi_scale_output = torch.cat(outputs, dim=-1)
        output = self.output_proj(multi_scale_output)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output

class MetalOptimizedAttention(nn.Module):
    """
    Optimized multi-head attention for Apple Silicon using Metal Performance Shaders (MPS).
    Features:
    1. Efficient memory usage through chunked computation
    2. Metal-accelerated matrix operations
    3. Gradient checkpointing for training larger models
    4. Adaptive attention patterns
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
        use_checkpointing: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_length = max_seq_length
        self.use_checkpointing = use_checkpointing
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
        # Create causal mask once
        mask = torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        
    def _chunk_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int = 128
    ) -> torch.Tensor:
        """
        Compute attention scores in memory-efficient chunks.
        """
        B, H, L, D = q.shape
        device = q.device
        
        output_chunks = []
        
        # Process in chunks
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            
            # Get current chunk
            q_chunk = q[:, :, chunk_start:chunk_end]
            
            # Compute attention scores for this chunk
            attn_weights = torch.matmul(q_chunk, k.transpose(-2, -1)) / self.scale
            
            # Apply causal mask
            if self.causal_mask is not None:
                mask_chunk = self.causal_mask[chunk_start:chunk_end, :L].to(device)
                attn_weights = attn_weights.masked_fill(mask_chunk, float('-inf'))
            
            # Compute softmax with better numerical stability
            max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]
            exp_weights = torch.exp(attn_weights - max_score)
            
            if self.causal_mask is not None:
                exp_weights = exp_weights.masked_fill(mask_chunk, 0.0)
            
            # Normalize
            attn_weights = exp_weights / (exp_weights.sum(dim=-1, keepdim=True) + 1e-6)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            chunk_output = torch.matmul(attn_weights, v)
            output_chunks.append(chunk_output)
        
        # Combine chunks
        output = torch.cat(output_chunks, dim=2)
        return output
    
    def _process_input(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input tensor into query, key, and value tensors.
        Applies Metal acceleration if available.
        """
        # Move to Metal device if available
        device = torch.device("mps") if torch.backends.mps.is_available() else x.device
        x = x.to(device)
        
        # Project to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            k = k.masked_fill(~key_padding_mask.unsqueeze(1).unsqueeze(-1), 0.0)
            v = v.masked_fill(~key_padding_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        
        return q, k, v
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional gradient checkpointing.
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            key_padding_mask: Optional mask for padded tokens
            need_weights: Whether to return attention weights
        Returns:
            output: Tensor of shape (batch_size, seq_len, dim)
            attn_weights: Optional attention weights if need_weights is True
        """
        # Process input
        if self.use_checkpointing and self.training:
            q, k, v = torch.utils.checkpoint.checkpoint(self._process_input, x, key_padding_mask)
        else:
            q, k, v = self._process_input(x, key_padding_mask)
        
        # Compute attention with chunking
        attn_output = self._chunk_attention(q, k, v)
        
        # Reshape and project output
        output = rearrange(attn_output, 'b h n d -> b n (h d)')
        output = self.o_proj(output)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        if need_weights:
            # Compute full attention weights if requested
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            if self.causal_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    self.causal_mask[:x.size(1), :x.size(1)].to(x.device),
                    float('-inf')
                )
            attn_weights = torch.softmax(attn_weights, dim=-1)
            return output, attn_weights
        
        return output, None
