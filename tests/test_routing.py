import pytest
import numpy as np
import torch
from odysee.routing import MultiModalRouter

def test_router_initialization():
    # Test valid initialization
    router = MultiModalRouter(routing_dim=1024, num_heads=8)
    assert router.routing_dim == 1024
    assert router.num_heads == 8
    
    # Test invalid initialization
    with pytest.raises(ValueError):
        MultiModalRouter(routing_dim=-1)
    with pytest.raises(ValueError):
        MultiModalRouter(routing_dim=1024, num_heads=0)

def test_route_text():
    router = MultiModalRouter(routing_dim=1024, num_heads=8)
    
    # Test with numpy array
    queries = np.random.randn(11, 1024).astype(np.float32)
    weights, indices = router.route_text(queries, batch_size=1, seq_len=11)
    assert weights.shape == (11, 8)
    assert indices.shape == (11, 8)
    
    # Test with torch tensor
    queries = torch.randn(11, 1024)
    weights, indices = router.route_text(queries, batch_size=1, seq_len=11)
    assert weights.shape == (11, 8)
    assert indices.shape == (11, 8)
    
    # Test invalid dimensions
    with pytest.raises(ValueError):
        queries = np.random.randn(11, 512).astype(np.float32)  # Wrong routing_dim
        router.route_text(queries)
        
    with pytest.raises(ValueError):
        queries = np.random.randn(11, 1024).astype(np.float32)
        router.route_text(queries, batch_size=2)  # Incompatible batch_size

def test_route_image():
    router = MultiModalRouter(routing_dim=1024, num_heads=8)
    
    # Test with numpy array (32x32 image = 4 patches)
    queries = np.random.randn(32, 32, 1024).astype(np.float32)
    weights, indices = router.route_image(queries)
    num_patches = ((32 + 15) // 16) * ((32 + 15) // 16)  # Should be 4 patches
    assert weights.shape == (num_patches, 8)
    assert indices.shape == (num_patches, 8)
    
    # Test with torch tensor
    queries = torch.randn(32, 32, 1024)
    weights, indices = router.route_image(queries)
    assert weights.shape == (num_patches, 8)
    assert indices.shape == (num_patches, 8)
    
    # Test with explicit image_size
    weights, indices = router.route_image(queries, image_size=(32, 32))
    assert weights.shape == (num_patches, 8)
    assert indices.shape == (num_patches, 8)
    
    # Test with non-square image (48x32 = 6 patches)
    queries = np.random.randn(48, 32, 1024).astype(np.float32)
    weights, indices = router.route_image(queries)
    num_patches = ((48 + 15) // 16) * ((32 + 15) // 16)  # Should be 6 patches
    assert weights.shape == (num_patches, 8)
    assert indices.shape == (num_patches, 8)
    
    # Test invalid dimensions
    with pytest.raises(ValueError):
        queries = np.random.randn(32, 32, 512).astype(np.float32)  # Wrong routing_dim
        router.route_image(queries)
        
    with pytest.raises(ValueError):
        queries = np.random.randn(32, 32, 1024).astype(np.float32)
        router.route_image(queries, image_size=(16, 16))  # Incompatible image_size
