import pytest
import numpy as np
from odysee.core.tensor import Tensor
from odysee.experts import (
    HierarchicalContextExpert, LongContextMoE, MLPExpert,
    ConvExpert, TransformerExpert
)

def test_hierarchical_expert_basic():
    expert = HierarchicalContextExpert(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_levels=3,
        chunk_size=512,
        overlap=64
    )
    
    # Test with small sequence
    x = Tensor(np.random.randn(2, 1000, 64))
    out = expert(x)
    assert out.shape == (2, 1000, 64)
    
    # Test with medium sequence
    x = Tensor(np.random.randn(2, 5000, 64))
    out = expert(x)
    assert out.shape == (2, 5000, 64)
    
    # Test cache reset
    expert.reset_cache()
    assert len(expert.cached_memories) == 0

def test_hierarchical_expert_long_context():
    expert = HierarchicalContextExpert(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_levels=5,
        chunk_size=512,
        overlap=64,
        max_context_length=1048576
    )
    
    # Test with very long sequence
    x = Tensor(np.random.randn(1, 100000, 64))
    out = expert(x)
    assert out.shape == (1, 100000, 64)
    
    # Test memory efficiency
    import psutil
    mem_before = psutil.Process().memory_info().rss
    x = Tensor(np.random.randn(1, 500000, 64))
    out = expert(x)
    mem_after = psutil.Process().memory_info().rss
    mem_used_gb = (mem_after - mem_before) / (1024 ** 3)
    assert mem_used_gb < 10, f"Memory usage too high: {mem_used_gb:.2f}GB"

def test_long_context_moe():
    model = LongContextMoE(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_experts_per_type=2,
        max_context_length=2097152  # 2M tokens
    )
    
    # Test routing behavior
    x = Tensor(np.random.randn(2, 1000, 64))
    out = model(x)
    assert out.shape == (2, 1000, 64)
    
    # Test expert selection
    routing_stats = model.get_routing_statistics()
    assert all(stats['usage'] > 0 for stats in routing_stats.values()), \
        "Some experts are not being used"
    
    # Test load balancing
    expert_loads = [stats['load'] for stats in routing_stats.values()]
    load_std = np.std(expert_loads)
    assert load_std < 0.1, f"Load balancing poor: std={load_std:.3f}"

@pytest.mark.parametrize("seq_len", [1000, 10000, 100000])
def test_hierarchical_expert_attention_patterns(seq_len):
    expert = HierarchicalContextExpert(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_levels=4,
        chunk_size=512,
        overlap=64
    )
    
    # Create input with clear patterns
    x = np.zeros((1, seq_len, 64))
    # Add periodic patterns
    for i in range(seq_len):
        x[0, i] = np.sin(2 * np.pi * i / 1000)
    
    x = Tensor(x)
    out = expert(x)
    
    # Check if long-range patterns are preserved
    input_pattern = x[0, :100].numpy()
    output_pattern = out[0, :100].numpy()
    correlation = np.corrcoef(input_pattern.mean(1), output_pattern.mean(1))[0, 1]
    assert correlation > 0.5, f"Long-range patterns not preserved: corr={correlation:.3f}"

def test_expert_types():
    experts = {
        'mlp': MLPExpert(64, 128, 64),
        'conv': ConvExpert(64, 128, 64),
        'transformer': TransformerExpert(64, 128, 64)
    }
    
    x = Tensor(np.random.randn(2, 100, 64))
    
    for name, expert in experts.items():
        out = expert(x)
        assert out.shape == (2, 100, 64), f"{name} expert output shape incorrect"

def test_memory_management():
    expert = HierarchicalContextExpert(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_levels=5,
        chunk_size=512,
        overlap=64,
        use_checkpointing=True
    )
    
    # Test gradient checkpointing
    x = Tensor(np.random.randn(1, 50000, 64), requires_grad=True)
    out = expert(x)
    loss = out.mean()
    loss.backward()
    
    assert x.grad is not None, "Gradient not computed"
    assert not np.isnan(x.grad.numpy()).any(), "NaN in gradients"

def test_expert_persistence():
    expert = HierarchicalContextExpert(
        input_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_levels=3,
        chunk_size=512,
        overlap=64
    )
    
    # Test state persistence
    x1 = Tensor(np.random.randn(2, 1000, 64))
    x2 = Tensor(np.random.randn(2, 1000, 64))
    
    out1 = expert(x1)
    cache_state = expert.cached_memories.copy()
    out2 = expert(x2)
    
    # Check if cache was updated
    assert any(not np.array_equal(cache_state[k].numpy(), 
                                expert.cached_memories[k].numpy())
              for k in cache_state), "Cache not updated"
    
    # Test cache influence
    expert.reset_cache()
    out2_fresh = expert(x2)
    
    # Outputs should be different with and without cache
    assert not np.array_equal(out2.numpy(), out2_fresh.numpy()), \
        "Cache has no effect on output"

if __name__ == '__main__':
    pytest.main([__file__])
