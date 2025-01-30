from typing import Optional, List, Tuple, Union, Callable
import numpy as np
from .tensor import Tensor
from .device import Device, get_default_device

class Parameter(Tensor):
    """Wrapper for parameters that need to be optimized"""
    def __init__(self, data: Union[np.ndarray, List, float], 
                 requires_grad: bool = True,
                 device: Optional[Device] = None):
        super().__init__(data, requires_grad=requires_grad, device=device)

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
        
    def __setattr__(self, name: str, value: any):
        if isinstance(value, (Parameter, Module)):
            if name.startswith('_'):
                super().__setattr__(name, value)
            else:
                self._parameters[name] = value
        else:
            super().__setattr__(name, value)
            
    def parameters(self) -> List[Parameter]:
        params = []
        for param in self._parameters.values():
            if isinstance(param, Parameter):
                params.append(param)
            elif isinstance(param, Module):
                params.extend(param.parameters())
        return params
    
    def eval(self):
        """Set the module in evaluation mode"""
        self.training = False
        for module in self._parameters.values():
            if isinstance(module, Module):
                module.eval()
                
    def train(self):
        """Set the module in training mode"""
        self.training = True
        for module in self._parameters.values():
            if isinstance(module, Module):
                module.train()
    
    def to(self, device: Device):
        """Move module to specified device"""
        for name, param in self._parameters.items():
            if isinstance(param, (Parameter, Tensor)):
                self._parameters[name] = param.to(device)
            elif isinstance(param, Module):
                param.to(device)
        return self

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: Optional[Device] = None):
        super().__init__()
        self.weight = Parameter(
            np.random.randn(in_features, out_features) / np.sqrt(in_features),
            device=device
        )
        if bias:
            self.bias = Parameter(np.zeros(out_features), device=device)
        else:
            self.bias = None
        
    def __call__(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True, device: Optional[Device] = None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
            
        self.stride = stride
        self.padding = padding
        
        # Initialize weights with Kaiming initialization
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        std = np.sqrt(2.0 / fan_in)
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, *kernel_size) * std,
            device=device
        )
        if bias:
            self.bias = Parameter(np.zeros(out_channels), device=device)
        else:
            self.bias = None
            
    def __call__(self, x: Tensor) -> Tensor:
        out = x.conv2d(self.weight, self.stride, self.padding)
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out

class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 device: Optional[Device] = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Parameter(np.ones(num_features), device=device)
        self.beta = Parameter(np.zeros(num_features), device=device)
        
        # Running statistics
        self.register_buffer('running_mean', np.zeros(num_features))
        self.register_buffer('running_var', np.ones(num_features))
        
    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = ((x - batch_mean.reshape(1, -1, 1, 1)) ** 2).mean(dim=(0, 2, 3))
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            
            # Normalize
            x_normalized = (x - batch_mean.reshape(1, -1, 1, 1)) / (batch_var.reshape(1, -1, 1, 1) + self.eps).sqrt()
        else:
            # Use running statistics
            x_normalized = (x - self.running_mean.reshape(1, -1, 1, 1)) / \
                         (self.running_var.reshape(1, -1, 1, 1) + self.eps).sqrt()
        
        return self.gamma.reshape(1, -1, 1, 1) * x_normalized + self.beta.reshape(1, -1, 1, 1)
    
    def register_buffer(self, name: str, tensor: np.ndarray):
        """Register a persistent buffer"""
        setattr(self, name, tensor)

class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5,
                 device: Optional[Device] = None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.gamma = Parameter(np.ones(normalized_shape), device=device)
        self.beta = Parameter(np.zeros(normalized_shape), device=device)
        
    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdims=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        
    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        
        mask = np.random.binomial(1, 1-self.p, x.shape)
        return x * mask / (1 - self.p)

# Activation functions
def relu(x: Tensor) -> Tensor:
    return x * (x > 0)

def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1 + (x * 0.7978845608028654).tanh() * (1 + 0.044715 * x * x))

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    exp_x = x.exp()
    return exp_x / exp_x.sum(dim=dim, keepdims=True)

# Optimizers
class Optimizer:
    def __init__(self, params: List[Parameter]):
        self.params = params
        
    def zero_grad(self):
        for param in self.params:
            param.grad = None
            
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params: List[Parameter], lr: float = 0.01, momentum: float = 0,
                 weight_decay: float = 0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in params]
        
    def step(self):
        for param, velocity in zip(self.params, self.velocities):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            velocity *= self.momentum
            velocity -= self.lr * grad
            param.data += velocity

class Adam(Optimizer):
    def __init__(self, params: List[Parameter], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        
        # Initialize momentum and velocity
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        
    def step(self):
        self.step_count += 1
        beta1, beta2 = self.betas
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            # Update momentum and velocity
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * grad * grad
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - beta2 ** self.step_count)
            
            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
