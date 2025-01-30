from typing import Optional, Tuple
import numpy as np

from .core.tensor import Tensor
from .core.nn import Module, Linear, LayerNorm, softmax, relu, Dropout
from .attention import MultiScaleAttention
from .routing import DynamicRouter
from .experts import (
    CrossLanguageExpertBase,
    QuantumFusionExpert,
    CompressedAttentionExpert,
    HierarchicalFusionExpert
)

class OdyseeLayer(Module):
    """
    A single layer of the Odysee architecture combining multi-scale attention,
    dynamic routing, and mixture of experts.
    """
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiScaleAttention(config)
        self.router = DynamicRouter(config)
        
        # Layer normalization
        self.ln1 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln3 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Mixture of Experts layer
        self.moe = LongContextMoE(
            input_dim=config.hidden_size,
            hidden_dim=config.ff_dim,
            output_dim=config.hidden_size,
            num_experts_per_type=config.num_experts,
            max_context_length=config.max_context_length
        )
        
        # Hierarchical feature processor
        self.hierarchical_processor = [
            Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_hierarchical_layers)
        ]
        
        # Output projection
        self.output_proj = Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = Dropout(config.dropout)
        
        # Gated feed-forward network
        if config.use_gated_ff:
            self.ff = [
                Linear(config.hidden_size, config.ff_dim),
                relu,
                Linear(config.ff_dim, config.hidden_size),
                self.dropout
            ]
            self.ff_gate = [
                Linear(config.hidden_size, config.ff_dim),
                softmax,
                Linear(config.ff_dim, config.hidden_size),
                self.dropout
            ]
        else:
            self.ff = [
                Linear(config.hidden_size, config.ff_dim),
                relu,
                self.dropout,
                Linear(config.ff_dim, config.hidden_size),
                self.dropout
            ]
            self.ff_gate = None
    
    def forward(
        self,
        x: Tensor,
        memory: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        # Multi-scale attention with memory
        attended = self.attention(self.ln1(x), memory, mask)
        x = x + self.dropout(attended)
        
        # Mixture of Experts processing
        moe_output = self.moe(self.ln2(x))
        x = x + self.dropout(moe_output)
        
        # Hierarchical feature processing
        hier_features = x
        for layer in self.hierarchical_processor:
            hier_features = relu(layer(hier_features))
            x = x + self.dropout(hier_features)
        
        # Update memory through routing
        if memory is not None:
            memory = self.router(self.ln3(x), memory)
        
        # Final output projection combining all features
        output = self.output_proj(
            Tensor.concat([x, hier_features], dim=-1)
        )
        
        # Gated feed-forward
        ff_output = x
        for layer in self.ff:
            if callable(layer):
                ff_output = layer(ff_output)
            else:
                ff_output = layer(ff_output)
        if self.ff_gate is not None:
            gate = x
            for layer in self.ff_gate:
                if callable(layer):
                    gate = layer(gate)
                else:
                    gate = layer(gate)
            ff_output = ff_output * gate
        
        return output + ff_output, memory

class OdyseeModel(Module):
    """
    The complete Odysee model architecture.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = Tensor(
            np.random.randn(config.vocab_size, config.hidden_size) / 
            np.sqrt(config.hidden_size),
            requires_grad=True
        )
        
        # Position embeddings
        if not config.use_relative_positions:
            self.pos_embedding = Tensor(
                np.zeros((1, config.max_sequence_length, config.hidden_size)),
                requires_grad=True
            )
        
        # Layers
        self.layers = [OdyseeLayer(config) for _ in range(config.num_layers)]
        
        # Memory slots
        self.memory = Tensor(
            np.random.randn(1, config.num_memory_slots, config.hidden_size),
            requires_grad=True
        )
        
        # Final normalization
        self.ln_f = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Cross-language experts
        self.experts = [
            QuantumFusionExpert(
                config.hidden_size,
                config.hidden_size,
                config.hidden_size
            ),
            CompressedAttentionExpert(
                config.hidden_size,
                config.hidden_size,
                config.hidden_size
            ),
            HierarchicalFusionExpert(
                config.hidden_size,
                config.hidden_size,
                config.hidden_size
            )
        ]
        
        # Expert routing
        self.router = Linear(config.hidden_size, len(self.experts))
        
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Get embeddings
        x = self.token_embedding[input_ids]
        
        if not self.config.use_relative_positions:
            x = x + self.pos_embedding[:, :x.shape[1]]
        
        # Initialize memory
        memory = self.memory
        
        # Process through layers
        for layer in self.layers:
            x, memory = layer(x, memory, attention_mask)
        
        # Route through experts
        router_logits = self.router(x)
        router_weights = softmax(router_logits, dim=-1)
        
        # Process through each expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            expert_outputs.append(expert_output * router_weights[..., i:i+1])
            
        # Combine expert outputs
        x = sum(expert_outputs)
        
        # Final normalization
        x = self.ln_f(x)
        return x
