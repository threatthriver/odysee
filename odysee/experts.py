from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from einops import rearrange, repeat

class ExpertBase(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, dim = x.shape
        
        # Convert to PyTorch tensor for processing
        x_torch = torch.from_numpy(x.data).to(torch.float32)
        if x.requires_grad:
            x_torch.requires_grad_(True)
            
        # Process through network
        output = x_torch
        
        # Convert back to our Tensor type
        result = Tensor(output.detach().numpy())
        if x.requires_grad:
            result.requires_grad = True
        return result

class MLPExpert(ExpertBase):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # Convert to PyTorch tensor
        x_torch = torch.from_numpy(x.data).to(torch.float32)
        if x.requires_grad:
            x_torch.requires_grad_(True)
        
        # Process through network
        out = self.net(x_torch)
        
        # Convert back to our Tensor type
        result = Tensor(out.detach().numpy())
        if x.requires_grad:
            result.requires_grad = True
        return result

class ConvExpert(ExpertBase):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # Convert to PyTorch tensor and preserve gradients
        x_torch = torch.from_numpy(x.data).to(torch.float32)
        if x.requires_grad:
            x_torch.requires_grad_(True)
        # Convert [batch, seq_len, channels] to [batch, channels, seq_len]
        x_torch = x_torch.transpose(1, 2)
        out = self.net(x_torch)
        # Convert back to [batch, seq_len, channels]
        out = out.transpose(1, 2)
        # Convert back to our Tensor type
        result = Tensor(out.detach().numpy())
        if x.requires_grad:
            result.requires_grad = True
        return result

class TransformerExpert(ExpertBase):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # Handle both PyTorch and custom Tensor inputs
        if isinstance(x, torch.Tensor):
            x_torch = x
        else:
            x_torch = x.data if isinstance(x.data, torch.Tensor) else torch.from_numpy(x.data)
        x_torch = x_torch.to(torch.float32)

        # Apply attention mechanism
        out = self.attention(x_torch)
        return Tensor(out, requires_grad=x.requires_grad)

class HierarchicalContextExpert(ExpertBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_levels: int = 3,
        chunk_size: int = 512,
        overlap: int = 64,
        max_context_length: Optional[int] = None,
        use_checkpointing: bool = False
    ):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.num_levels = num_levels
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_context_length = max_context_length or (chunk_size * (2 ** (num_levels - 1)))
        self.use_checkpointing = use_checkpointing
        
        # Create experts for each level
        self.local_experts = nn.ModuleList([
            TransformerExpert(input_dim, hidden_dim, output_dim)
            for _ in range(num_levels)
        ])
        
        # Cross attention between levels
        self.cross_experts = nn.ModuleList([
            TransformerExpert(output_dim, hidden_dim, output_dim)
            for _ in range(num_levels - 1)
        ])
        
        # Memory cache for each level
        self.cached_memories = [[] for _ in range(num_levels)]
        self.reset_caches()
        
    def reset_caches(self):
        self.memory_cache = [[] for _ in range(self.num_levels)]
        self.cached_memories = [[] for _ in range(self.num_levels)]
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, dim = x.shape
        chunk_size = min(self.chunk_size, seq_len)
        
        # Process through levels
        current = x
        for level in range(self.num_levels):
            # Process current level
            processed, memories = self._process_level(current, level, chunk_size)
            self.cached_memories[level] = memories
            
            # Update chunk size for next level
            chunk_size *= 2
            current = processed
        
        return current
    
    def _process_level(
        self,
        x: Tensor,
        level: int,
        chunk_size: int
    ) -> Tuple[Tensor, List[Tensor]]:
        B, L, D = x.shape
        
        # Convert to PyTorch tensor for processing
        x_torch = torch.from_numpy(x.data).to(torch.float32)
        if x.requires_grad:
            x_torch.requires_grad_(True)
        
        # Split into chunks with overlap
        chunks = []
        memories = []
        
        for i in range(0, L, chunk_size - self.overlap):
            end = min(i + chunk_size, L)
            chunk = x_torch[:, i:end]
            
            # Process chunk
            chunk_tensor = Tensor(chunk.detach().numpy())
            processed = self.local_experts[level](chunk_tensor)
            processed_torch = torch.from_numpy(processed.data).to(torch.float32)
            if processed.requires_grad:
                processed_torch.requires_grad_(True)
            chunks.append(processed_torch)
            
            # Extract memory
            if end < L:  # Not the last chunk
                memory = chunks[-1][:, -self.overlap:]
                memories.append(memory)
        
        # Combine chunks
        output = torch.cat(chunks, dim=1)
        if output.shape[1] > L:  # Trim overlap
            output = output[:, :L]
            
        result = Tensor(output.detach().numpy())
        if x.requires_grad:
            result.requires_grad = True
        return result, memories

class LongContextMoE(ExpertBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts_per_type: int = 4,
        max_context_length: Optional[int] = None
    ):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.max_context_length = max_context_length or 4194304  # Default 4M tokens
        self.routing_stats = {}
        
        # Create different types of experts
        self.experts = nn.ModuleDict({
            'mlp': nn.ModuleList([MLPExpert(input_dim, hidden_dim, output_dim) 
                                for _ in range(num_experts_per_type)]),
            'conv': nn.ModuleList([ConvExpert(input_dim, hidden_dim, output_dim)
                                for _ in range(num_experts_per_type)]),
            'transformer': nn.ModuleList([TransformerExpert(input_dim, hidden_dim, output_dim)
                                      for _ in range(num_experts_per_type)]),
            'hierarchical': nn.ModuleList([HierarchicalContextExpert(input_dim, hidden_dim, output_dim)
                                       for _ in range(num_experts_per_type)])
        })
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.experts) * num_experts_per_type)
        )
        
    def get_routing_statistics(self) -> Dict:
        """Returns the routing statistics for analysis"""
        return self.routing_stats
        
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, dim = x.shape
        
        # Convert to PyTorch tensor with correct dtype
        x_torch = torch.from_numpy(x.data).to(torch.float32)
        if x.requires_grad:
            x_torch.requires_grad_(True)
        
        # Get routing probabilities
        route_logits = self.router(x_torch.mean(dim=1))  # [B, num_experts_total]
        route_probs = torch.softmax(route_logits, dim=-1)
        
        # Split routing probabilities by expert type
        num_expert_types = len(self.experts)
        route_probs = route_probs.view(batch_size, num_expert_types, -1)
        
        # Update routing statistics
        with torch.no_grad():
            self.routing_stats = {
                expert_type: probs.mean().item()
                for expert_type, probs in zip(self.experts.keys(), route_probs.mean(0))
            }
        
        # Process input through each expert type
        outputs = []
        for i, (expert_type, experts) in enumerate(self.experts.items()):
            expert_probs = route_probs[:, i]  # [B, num_experts_per_type]
            
            # Process through each expert of this type
            expert_outputs = []
            for j, expert in enumerate(experts):
                out = expert(x)
                out_torch = torch.from_numpy(out.data).to(torch.float32)
                if out.requires_grad:
                    out_torch.requires_grad_(True)
                expert_outputs.append(out_torch * expert_probs[:, j].view(batch_size, 1, 1))
            
            # Combine experts of same type
            type_output = sum(expert_outputs)
            outputs.append(type_output)
        
        # Combine all expert types
        final_output = sum(outputs)
        result = Tensor(final_output.detach().numpy())
        if x.requires_grad:
            result.requires_grad = True
        return result

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np
from einops import rearrange, repeat

# Temporarily remove rust experts import until module is available
# from .rust_experts import QuantumInspiredExpert as RustQuantumExpert
# Temporarily remove rust experts import until module is available
# from .rust_experts import NeuralCompressionExpert as RustCompressionExpert

class MetalAcceleratedAttention(nn.Module):
    """Efficient attention implementation optimized for Apple Metal"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 1024
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Create causal mask once
        mask = torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool()
        self.register_buffer("mask", mask)
        
    def forward(self, x: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        B, L, D = x.shape
        
        # Move to Metal device if available
        device = torch.device("mps") if torch.backends.mps.is_available() else x.device
        x = x.to(device)
        
        # Project to q, k, v
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (three h d) -> three b h n d', three=3, h=self.num_heads)
        
        # Compute attention scores with memory-efficient implementation
        scale = self.scaling
        
        # Split processing into chunks for memory efficiency
        chunk_size = min(L, 256)  # Process 256 tokens at a time
        output_chunks = []
        
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            
            # Get current chunk
            q_chunk = q[:, :, chunk_start:chunk_end]
            
            # Compute attention scores for this chunk
            attn_weights = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
            
            # Apply causal mask
            mask_chunk = self.mask[chunk_start:chunk_end, :L]
            attn_weights = attn_weights.masked_fill(mask_chunk, float('-inf'))
            
            # Compute softmax with better numerical stability
            max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]
            exp_weights = torch.exp(attn_weights - max_score)
            exp_weights = exp_weights.masked_fill(mask_chunk, 0.0)
            
            # Normalize
            attn_weights = exp_weights / (exp_weights.sum(dim=-1, keepdim=True) + 1e-6)
            
            # Apply attention to values
            chunk_output = torch.matmul(attn_weights, v)
            output_chunks.append(chunk_output)
        
        # Combine chunks
        output = torch.cat(output_chunks, dim=2)
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        # Project back
        output = self.proj(output)
        output = self.dropout(output)
        
        return output.to(x.device)

class QuantumFusionExpert(nn.Module):
    """Expert that combines quantum-inspired processing with neural networks"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_qubits: int = 8
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Quantum-inspired neural layers
        self.quantum_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(num_qubits)
        ])
        
        # Neural layers
        self.pre_quantum = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.post_quantum = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Quantum mixing parameters
        self.quantum_mixer = nn.Parameter(torch.randn(3, hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-process input
        x = self.pre_quantum(x)
        
        # Process through quantum-inspired layers
        quantum_states = [layer(x) for layer in self.quantum_layers]
        quantum_state = torch.stack(quantum_states, dim=-2)
        
        # Apply quantum mixing
        if torch.backends.mps.is_available():
            quantum_state = quantum_state.to("mps")
            self.quantum_mixer.data = self.quantum_mixer.data.to("mps")
            
        mixed = torch.einsum('b...d,qd->b...qd', quantum_state, self.quantum_mixer)
        mixed = mixed.mean(dim=-2)
        
        # Post-process
        output = self.post_quantum(mixed)
        
        return output

class CompressedAttentionExpert(nn.Module):
    """Expert that combines neural compression with efficient attention"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        compression_ratio: float = 0.25
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Neural compression layers
        compressed_dim = int(hidden_dim * compression_ratio)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, compressed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Metal-accelerated attention
        self.attention = MetalAcceleratedAttention(
            dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compress input using neural layers
        compressed = self.encoder(x)
        compressed = self.decoder(compressed)
        
        # Process with attention
        if torch.backends.mps.is_available():
            compressed = compressed.to("mps")
            
        attention_output = self.attention(compressed)
        
        # Project to output dimension
        output = self.output_proj(attention_output)
        
        return output

class HierarchicalFusionExpert(nn.Module):
    """Expert that combines quantum and compressed attention hierarchically"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_levels: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        
        # Create experts for each level
        self.quantum_experts = nn.ModuleList([
            QuantumFusionExpert(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                hidden_dim
            ) for i in range(num_levels)
        ])
        
        self.attention_experts = nn.ModuleList([
            CompressedAttentionExpert(
                hidden_dim,
                hidden_dim,
                hidden_dim if i < num_levels-1 else output_dim
            ) for i in range(num_levels)
        ])
        
        # Level mixing
        self.level_mixing = nn.Parameter(torch.randn(num_levels, 2))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Process through levels
        current = x
        level_outputs = []
        
        for i in range(self.num_levels):
            # Process through both expert types
            quantum_output = self.quantum_experts[i](current)
            attention_output = self.attention_experts[i](current)
            
            # Mix outputs for this level
            if torch.backends.mps.is_available():
                quantum_output = quantum_output.to("mps")
                attention_output = attention_output.to("mps")
                self.level_mixing.data = self.level_mixing.data.to("mps")
                
            level_weights = F.softmax(self.level_mixing[i], dim=-1)
            level_output = (
                level_weights[0] * quantum_output +
                level_weights[1] * attention_output
            )
            
            level_outputs.append(level_output)
            current = level_output
            
        # Combine all levels with residual connections
        output = sum(level_outputs)
        
        return output
