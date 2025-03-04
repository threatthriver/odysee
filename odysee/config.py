from dataclasses import dataclass
from typing import Optional

@dataclass
class RoutingConfig:
    """Configuration for dynamic routing"""
    routing_dim: int
    num_heads: int
    max_context_length: int
    use_flash_attention: bool = True
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    hidden_act: str = "gelu"
    intermediate_size: Optional[int] = None
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.routing_dim

@dataclass
class OdyseeConfig:
    """Configuration for Odysee model"""
    
    # Model dimensions
    input_dim: int
    hidden_dim: int
    output_dim: int
    
    # Attention parameters
    num_heads: int = 8
    dropout: float = 0.1
    
    # Expert parameters
    num_experts_per_type: int = 4
    max_context_length: Optional[int] = None  # Default 4M in LongContextMoE
    
    # Memory management
    use_checkpointing: bool = False
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.output_dim > 0, "output_dim must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"
        assert self.num_experts_per_type > 0, "num_experts_per_type must be positive"
        
        if self.max_context_length is not None:
            assert self.max_context_length > 0, "max_context_length must be positive"
