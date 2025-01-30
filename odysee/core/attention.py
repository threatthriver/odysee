from typing import Optional, Tuple
import numpy as np
from .tensor import Tensor
from .nn import Module, Parameter, Linear, softmax

class MultiHeadAttention(Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear layers for Q, K, V projections
        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, hidden_size)
        self.v_proj = Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = Linear(hidden_size, hidden_size)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """Split heads and transpose to get [batch, heads, seq, head_dim]"""
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def _merge_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """Transpose and merge heads to get [batch, seq, hidden]"""
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, -1, self.hidden_size)
    
    def forward(self, 
               query: Tensor,
               key: Optional[Tensor] = None,
               value: Optional[Tensor] = None,
               mask: Optional[Tensor] = None) -> Tensor:
        batch_size = query.shape[0]
        
        # Default to self-attention if key/value not provided
        key = query if key is None else key
        value = query if value is None else value
        
        # Project and split heads
        q = self._split_heads(self.q_proj(query), batch_size)  # [B, H, Tq, D]
        k = self._split_heads(self.k_proj(key), batch_size)    # [B, H, Tk, D]
        v = self._split_heads(self.v_proj(value), batch_size)  # [B, H, Tv, D]
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # [B, H, Tq, Tk]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = softmax(scores, dim=-1)
        attn_output = attn_weights @ v  # [B, H, Tq, D]
        
        # Merge heads and project
        output = self._merge_heads(attn_output, batch_size)
        return self.out_proj(output)

class RelativePositionalEncoding(Module):
    def __init__(self, hidden_size: int, max_seq_len: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Create relative position encoding matrix
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, hidden_size, 2) * -(np.log(10000.0) / hidden_size))
        
        pe = np.zeros((max_seq_len, hidden_size))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """Add relative positional encodings to the input"""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]

class LocalAttention(Module):
    """Attention that only attends to a local window around each position"""
    def __init__(self, hidden_size: int, window_size: int, num_heads: int = 8):
        super().__init__()
        self.window_size = window_size
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        
    def _create_local_mask(self, seq_len: int) -> Tensor:
        """Create mask for local attention window"""
        mask = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        return Tensor(mask)
    
    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        mask = self._create_local_mask(seq_len)
        return self.attention(x, mask=mask)

class LinearAttention(Module):
    """Linear attention with O(N) complexity instead of O(N^2)"""
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, hidden_size)
        self.v_proj = Linear(hidden_size, hidden_size)
        self.out_proj = Linear(hidden_size, hidden_size)
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply feature map (ELU + 1)
        q = q * (q > 0) + 1
        k = k * (k > 0) + 1
        
        # Linear attention
        kv = (k.transpose(1, 2) @ v.transpose(1, 2)).transpose(1, 2)  # [B, H, D, D]
        qkv = q @ kv  # [B, N, H, D]
        
        # Normalize
        normalizer = q @ k.sum(dim=1).unsqueeze(-1)  # [B, N, H, 1]
        output = (qkv / (normalizer + 1e-6)).reshape(batch_size, seq_len, -1)
        
        return self.out_proj(output)
