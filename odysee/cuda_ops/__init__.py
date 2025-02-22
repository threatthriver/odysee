"""
CUDA/CPU operations for Odysee
"""
import torch
import torch.nn.functional as F
import logging
import os
import platform
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Check platform and device availability
IS_MACOS = platform.system() == 'Darwin'
IS_ARM = platform.machine() == 'arm64'
# Check CUDA environment and version
CUDA_HOME = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
HAS_CUDA = torch.cuda.is_available() and CUDA_HOME is not None

# Get CUDA version if available
CUDA_VERSION = None
if HAS_CUDA:
    try:
        CUDA_VERSION = torch.version.cuda
        logger.info(f"CUDA version {CUDA_VERSION} detected")
    except:
        logger.warning("Could not detect CUDA version")
HAS_MPS = IS_MACOS and IS_ARM and torch.backends.mps.is_available()

def get_device():
    """Get the best available device for computation"""
    if HAS_CUDA:
        return torch.device('cuda')
    elif HAS_MPS:
        return torch.device('mps')
    else:
        return torch.device('cpu')

def to_device(tensor):
    """Move tensor to the best available device"""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor.to(get_device())

def to_numpy(tensor):
    """Convert tensor to numpy array, handling different devices properly"""
    if isinstance(tensor, torch.Tensor):
        if tensor.device.type == 'mps':
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    return tensor

class QuantumPhaseEncoder(torch.nn.Module):
    """Quantum phase encoding implementation"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phase_embedding = torch.nn.Parameter(torch.randn(hidden_dim) / hidden_dim ** 0.5)
        
    def forward(self, x):
        # Apply quantum phase encoding
        phase = torch.sin(x * self.phase_embedding[None, None, :])
        amplitude = torch.cos(x * self.phase_embedding[None, None, :])
        return phase * amplitude

class FlashAttention(torch.nn.Module):
    """Optimized attention implementation with improved memory efficiency"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        
        # Initialize weights with proper scaling
        std = self.head_dim ** -0.5
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize parameters with proper scaling
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            torch.nn.init.normal_(proj.weight, mean=0.0, std=std)
            if proj.bias is not None:
                torch.nn.init.zeros_(proj.bias)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Project and scale inputs
        q = self.q_proj(query) * self.scale
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention with memory-efficient ordering
        q = q.contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with improved numerical stability
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply attention with memory-efficient implementation
        if HAS_CUDA:
            try:
                from flash_attn.flash_attention import FlashAttention
                flash = FlashAttention(
                    softmax_scale=1.0,  # Scale already applied to query
                    attention_dropout=self.dropout if self.training else 0.0,
                    device=query.device
                )
                output = flash(q, k, v, key_padding_mask=mask)[0]
            except ImportError:
                attn = torch.softmax(scores, dim=-1)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                output = torch.matmul(attn, v)
        else:
            # Optimized attention for CPU/MPS
            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            output = torch.matmul(attn, v)
        
        # Reshape and project output with proper memory layout
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.out_proj(output)

# Initialize models
_quantum_encoder = None
_flash_attention = None

def _get_quantum_encoder(hidden_dim):
    global _quantum_encoder
    if _quantum_encoder is None:
        _quantum_encoder = QuantumPhaseEncoder(hidden_dim).to(get_device())
    return _quantum_encoder

def _get_flash_attention(hidden_dim, num_heads):
    global _flash_attention
    if _flash_attention is None:
        _flash_attention = FlashAttention(hidden_dim, num_heads).to(get_device())
    return _flash_attention

def quantum_phase_encoding(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply quantum phase encoding to input tensor.
    Automatically uses the best available device (CUDA/MPS/CPU).
    
    Args:
        input_tensor: Input tensor of shape (batch_size, seq_len, hidden_dim)
        
    Returns:
        Encoded tensor of the same shape
    """
    input_tensor = to_device(input_tensor)
    encoder = _get_quantum_encoder(input_tensor.size(-1))
    output = encoder(input_tensor)
    return output

def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply attention mechanism with optimizations for available hardware.
    Uses flash attention on CUDA, optimized attention on MPS, and standard attention on CPU.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, hidden_dim)
        key: Key tensor of shape (batch_size, seq_len, hidden_dim)
        value: Value tensor of shape (batch_size, seq_len, hidden_dim)
        num_heads: Number of attention heads
        mask: Optional attention mask
        
    Returns:
        Output tensor of shape (batch_size, seq_len, hidden_dim)
    """
    # Move inputs to device
    device = get_device()
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # Get flash attention module
    if HAS_CUDA:
        try:
            from flash_attn.flash_attention import FlashAttention
            flash = FlashAttention(
                softmax_scale=None,
                attention_dropout=0.0,  # Handled separately
                device=device
            )
            # Reshape for flash attention
            batch_size, seq_len, hidden_dim = query.shape
            head_dim = hidden_dim // num_heads
            query = query.view(batch_size, seq_len, num_heads, head_dim)
            key = key.view(batch_size, seq_len, num_heads, head_dim)
            value = value.view(batch_size, seq_len, num_heads, head_dim)
            
            # Apply flash attention
            with torch.cuda.amp.autocast(enabled=True):
                output = flash(query, key, value, key_padding_mask=mask)[0]
                
            # Reshape output
            output = output.view(batch_size, seq_len, hidden_dim)
            
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Flash attention unavailable: {e}. Falling back to standard attention.")
            output = standard_attention(query, key, value, mask, num_heads)
    else:
        output = standard_attention(query, key, value, mask, num_heads)
    
    return output

def standard_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    num_heads: int
) -> torch.Tensor:
    """Standard scaled dot-product attention implementation."""
    batch_size, seq_len, hidden_dim = query.shape
    head_dim = hidden_dim // num_heads
    
    # Reshape for multi-head attention
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    
    # Reshape output
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
    return output
