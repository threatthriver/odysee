import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from .device import Device, get_default_device

@dataclass
class Device:
    """Represents a computation device (CPU/GPU)"""
    name: str
    type: str

class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, torch.Tensor):
            self.data = data.detach().numpy()
        else:
            self.data = np.asarray(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._torch_tensor = None

    @property
    def shape(self):
        return self.data.shape

    def to_torch(self):
        if self._torch_tensor is None:
            self._torch_tensor = torch.from_numpy(self.data)
            if self.requires_grad:
                self._torch_tensor.requires_grad_(True)
        return self._torch_tensor

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def to(self, dtype):
        if isinstance(dtype, torch.dtype):
            return Tensor(self.data.astype(self._numpy_dtype(dtype)), requires_grad=self.requires_grad)
        return self

    def _numpy_dtype(self, torch_dtype):
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64
        }
        return dtype_map.get(torch_dtype, np.float32)

    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=self.requires_grad)

    def numpy(self):
        return self.data
        self.device = device or get_default_device()
        if isinstance(data, np.ndarray):
            self.data = self.device.transfer_array(data)
        elif isinstance(data, torch.Tensor):
            self.data = self.device.transfer_array(data.detach().cpu().numpy())
        else:
            self.data = self.device.transfer_array(np.array(data))
        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None
        self._backward_fn = lambda: None
        self._prev = set()
        
    @property
    def shape(self) -> Tuple:
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def to(self, device: Device) -> 'Tensor':
        """Move tensor to another device"""
        if device.type == self.device.type:
            return self
        new_data = device.transfer_array(self.data)
        return Tensor(new_data, self.requires_grad, device)
    
    def detach(self) -> 'Tensor':
        """Create a new tensor detached from the computation graph"""
        return Tensor(self.data, requires_grad=False, device=self.device)
    
    def numpy(self) -> np.ndarray:
        """Convert tensor to numpy array"""
        if self.device.is_cuda:
            return self.data.get()
        return self.data
    
    def backward(self, grad: Optional['Tensor'] = None):
        if grad is None:
            grad = Tensor(np.ones_like(self.data), device=self.device)
            
        self.grad = grad
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                topo.append(v)
                
        build_topo(self)
        
        for node in reversed(topo):
            node._backward_fn()
    
    # Basic arithmetic operations
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data + other.data, 
                    requires_grad=self.requires_grad or other.requires_grad,
                    device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad = Tensor(out.grad.data, device=self.device)
            if other.requires_grad:
                other.grad = Tensor(out.grad.data, device=self.device)
                
        out._backward_fn = _backward
        out._prev = {self, other}
        return out
    
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(self.data * other.data, 
                    requires_grad=self.requires_grad or other.requires_grad,
                    device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad = Tensor(other.data * out.grad.data, device=self.device)
            if other.requires_grad:
                other.grad = Tensor(self.data * out.grad.data, device=self.device)
                
        out._backward_fn = _backward
        out._prev = {self, other}
        return out
    
    # Advanced operations
    def matmul(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(self.device.xp.matmul(self.data, other.data), 
                    requires_grad=self.requires_grad or other.requires_grad,
                    device=self.device)
        
        def _backward():
            if self.requires_grad:
                self.grad = Tensor(self.device.xp.matmul(out.grad.data, other.data.T),
                                 device=self.device)
            if other.requires_grad:
                other.grad = Tensor(self.device.xp.matmul(self.data.T, out.grad.data),
                                  device=self.device)
                
        out._backward_fn = _backward
        out._prev = {self, other}
        return out
    
    def conv2d(self, weight: 'Tensor', stride: Tuple[int, int] = (1, 1), 
               padding: Tuple[int, int] = (0, 0)) -> 'Tensor':
        """2D convolution operation"""
        # Implementation using im2col for efficiency
        N, C, H, W = self.shape
        F, C, HH, WW = weight.shape
        
        # Calculate output dimensions
        H_out = (H + 2 * padding[0] - HH) // stride[0] + 1
        W_out = (W + 2 * padding[1] - WW) // stride[1] + 1
        
        # Implement im2col
        x_cols = self._im2col(self.data, HH, WW, stride, padding)
        w_cols = weight.data.reshape(F, -1)
        
        # Compute convolution
        out_data = self.device.xp.matmul(w_cols, x_cols).reshape(F, H_out, W_out, N).transpose(3, 0, 1, 2)
        out = Tensor(out_data, requires_grad=self.requires_grad or weight.requires_grad, device=self.device)
        
        def _backward():
            if self.requires_grad:
                # Implement backward pass for input
                pass
            if weight.requires_grad:
                # Implement backward pass for weight
                pass
                
        out._backward_fn = _backward
        out._prev = {self, weight}
        return out
    
    def _im2col(self, x: Any, HH: int, WW: int, stride: Tuple[int, int], 
                padding: Tuple[int, int]) -> Any:
        """Convert image tensor to column matrix for efficient convolution"""
        N, C, H, W = x.shape
        
        # Add padding
        p_h, p_w = padding
        x_padded = self.device.xp.pad(x, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), mode='constant')
        
        # Calculate output dimensions
        H_out = (H + 2 * p_h - HH) // stride[0] + 1
        W_out = (W + 2 * p_w - WW) // stride[1] + 1
        
        # Create column matrix
        cols = self.device.xp.zeros((C * HH * WW, H_out * W_out * N))
        
        # Fill column matrix (this can be optimized further)
        for i in range(H_out):
            for j in range(W_out):
                col = x_padded[:, :, i*stride[0]:i*stride[0]+HH, j*stride[1]:j*stride[1]+WW].reshape(-1)
                cols[:, i*W_out + j] = col
                
        return cols
    
    def max_pool2d(self, kernel_size: Tuple[int, int], stride: Optional[Tuple[int, int]] = None) -> 'Tensor':
        """2D max pooling operation"""
        if stride is None:
            stride = kernel_size
            
        N, C, H, W = self.shape
        HH, WW = kernel_size
        
        # Calculate output dimensions
        H_out = (H - HH) // stride[0] + 1
        W_out = (W - WW) // stride[1] + 1
        
        out_data = self.device.xp.zeros((N, C, H_out, W_out))
        
        # Naive implementation (can be optimized)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        window = self.data[n, c, 
                                        i*stride[0]:i*stride[0]+HH, 
                                        j*stride[1]:j*stride[1]+WW]
                        out_data[n, c, i, j] = self.device.xp.max(window)
        
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device)
        
        def _backward():
            if self.requires_grad:
                # Implement max pool backward pass
                pass
                
        out._backward_fn = _backward
        out._prev = {self}
        return out
    
    @staticmethod
    def cat(tensors: List['Tensor'], dim: int = 0) -> 'Tensor':
        """Concatenate tensors along specified dimension"""
        device = tensors[0].device
        data = device.xp.concatenate([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        return Tensor(data, requires_grad=requires_grad, device=device)
    
    def reshape(self, *shape) -> 'Tensor':
        """Reshape tensor"""
        return Tensor(self.data.reshape(*shape), 
                     requires_grad=self.requires_grad, 
                     device=self.device)
    
    def mean(self, dim: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Compute mean along specified dimension"""
        out_data = self.device.xp.mean(self.data, axis=dim, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device)
        
        def _backward():
            if self.requires_grad:
                if dim is None:
                    grad_shape = self.shape
                else:
                    grad_shape = list(self.shape)
                    if not keepdims:
                        grad_shape.pop(dim)
                scale = self.device.xp.ones(grad_shape) / (self.data.size if dim is None else self.shape[dim])
                self.grad = Tensor(scale * out.grad.data, device=self.device)
                
        out._backward_fn = _backward
        out._prev = {self}
        return out
