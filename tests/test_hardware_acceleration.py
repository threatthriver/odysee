import pytest
import torch
import numpy as np
from odysee.cuda_ops import get_device, to_device, quantum_phase_encoding, flash_attention

def test_device_detection():
    device = get_device()
    assert isinstance(device, torch.device)
    
    # Test device string format
    device_str = device.type
    assert device_str in ['cuda', 'mps', 'cpu']
    
    if device_str == 'cuda':
        assert torch.cuda.is_available()
    elif device_str == 'mps':
        assert torch.backends.mps.is_available()

def test_tensor_device_movement():
    # Create test tensor with float32 dtype
    x = torch.randn(10, 10, dtype=torch.float32)
    device_tensor = to_device(x)
    
    # Verify device movement
    assert device_tensor.device.type == get_device().type
    
    # Test numpy array conversion with float32
    np_array = np.random.randn(10, 10).astype(np.float32)
    device_tensor = to_device(np_array)
    assert device_tensor.device.type == get_device().type

def test_quantum_phase_encoding():
    # Test input with correct shape and dtype
    x = torch.randn(2, 32, 64, dtype=torch.float32)
    
    # Apply quantum encoding
    encoded = quantum_phase_encoding(x)
    
    # Verify output
    assert encoded.shape == (2, 32, 8)  # Output shape matches num_qubits
    assert encoded.device.type == get_device().type

def test_flash_attention():
    # Test inputs
    batch_size = 2
    seq_len = 32
    hidden_dim = 64
    num_heads = 8
    
    query = torch.randn(batch_size, seq_len, hidden_dim)
    key = torch.randn(batch_size, seq_len, hidden_dim)
    value = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Apply attention
    output = flash_attention(query, key, value, num_heads)
    
    # Verify output
    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert output.device == get_device()

def test_memory_efficiency():
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Run memory-intensive operation
    x = torch.randn(1000, 1000)
    y = to_device(x)
    del y
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    final_memory = process.memory_info().rss
    memory_diff = final_memory - initial_memory
    
    # Verify reasonable memory usage (less than 1GB)
    assert memory_diff < 1e9

def test_hardware_specific_optimizations():
    device = get_device()
    device_str = str(device)
    
    if device_str == 'cuda':
        # Test CUDA-specific optimizations
        assert torch.cuda.is_available()
        x = torch.randn(100, 100, device='cuda')
        assert x.is_cuda
        
    elif device_str == 'mps':
        # Test MPS-specific optimizations
        assert torch.backends.mps.is_available()
        x = torch.randn(100, 100, device='mps')
        assert x.device.type == 'mps'
        
    else:
        # Test CPU optimizations
        x = torch.randn(100, 100)
        assert x.device.type == 'cpu'