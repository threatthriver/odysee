"""
CUDA/CPU operations for Odysee
"""
import torch
import torch.nn.functional as F
import logging
import os
import platform

logger = logging.getLogger(__name__)

# Check platform and device availability
IS_MACOS = platform.system() == 'Darwin'
IS_ARM = platform.machine() == 'arm64'
HAS_CUDA = torch.cuda.is_available() and os.environ.get('CUDA_HOME') is not None
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
    """Optimized attention implementation"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        # Initialize weights
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention
        if HAS_CUDA:
            # Use CUDA optimized attention if available
            try:
                from flash_attn.flash_attention import FlashAttention
                flash = FlashAttention(softmax_scale=None, attention_dropout=self.dropout)
                output = flash(q, k, v)[0]
            except ImportError:
                attn = F.softmax(scores, dim=-1)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                output = torch.matmul(attn, v)
        else:
            # Standard attention for CPU/MPS
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            output = torch.matmul(attn, v)
        
        # Reshape and project output
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
    num_heads: int
) -> torch.Tensor:
    """
    Apply attention mechanism with optimizations for available hardware.
    Uses flash attention on CUDA, optimized attention on MPS, and standard attention on CPU.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, hidden_dim)
        key: Key tensor of shape (batch_size, seq_len, hidden_dim)
        value: Value tensor of shape (batch_size, seq_len, hidden_dim)
        num_heads: Number of attention heads
        
    Returns:
        Output tensor of shape (batch_size, seq_len, hidden_dim)
    """
    # Move inputs to device
    query = to_device(query)
    key = to_device(key)
    value = to_device(value)
    
    # Get attention module
    attention = _get_flash_attention(query.size(-1), num_heads)
    
    output = attention(query, key, value)
    
    # Convert to numpy if needed for routing
    if hasattr(output, 'requires_grad') and output.requires_grad:
        output = output.detach()
    
    return output
