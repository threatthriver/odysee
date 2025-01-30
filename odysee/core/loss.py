from typing import Optional
import numpy as np
from .tensor import Tensor

class Loss:
    """Base class for all loss functions"""
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

class MSELoss(Loss):
    """Mean Squared Error Loss"""
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target) ** 2).mean()

class CrossEntropyLoss(Loss):
    """Cross Entropy Loss with optional label smoothing"""
    def __init__(self, label_smoothing: float = 0.0):
        self.label_smoothing = label_smoothing
        
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        # Apply log softmax
        log_probs = pred - pred.exp().sum(dim=-1, keepdims=True).log()
        
        if self.label_smoothing > 0:
            n_classes = pred.shape[-1]
            smooth_target = target * (1 - self.label_smoothing) + self.label_smoothing / n_classes
            return -(smooth_target * log_probs).sum(dim=-1).mean()
        else:
            return -(target * log_probs).sum(dim=-1).mean()

class BCEWithLogitsLoss(Loss):
    """Binary Cross Entropy with Logits Loss"""
    def __init__(self, pos_weight: Optional[Tensor] = None):
        self.pos_weight = pos_weight
        
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.pos_weight is not None:
            loss = self.pos_weight * target * (-pred).relu() + \
                   (1 - target) * (-(-pred)).relu() + \
                   pred.abs().log1p()
        else:
            loss = target * (-pred).relu() + \
                   (1 - target) * (-(-pred)).relu() + \
                   pred.abs().log1p()
        return loss.mean()

class FocalLoss(Loss):
    """Focal Loss for dealing with class imbalance"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        # Apply sigmoid
        probs = 1 / (1 + (-pred).exp())
        
        # Calculate focal weights
        pt = target * probs + (1 - target) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        # Calculate loss
        loss = -alpha_t * focal_weight * (
            target * pred.log_sigmoid() + (1 - target) * (-pred.log_sigmoid())
        )
        return loss.mean()

class HuberLoss(Loss):
    """Huber Loss (L1 for large errors, L2 for small errors)"""
    def __init__(self, delta: float = 1.0):
        self.delta = delta
        
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        quadratic = 0.5 * diff ** 2
        linear = self.delta * abs_diff - 0.5 * self.delta ** 2
        return (abs_diff <= self.delta) * quadratic + \
               (abs_diff > self.delta) * linear

class KLDivLoss(Loss):
    """Kullback-Leibler Divergence Loss"""
    def __init__(self, reduction: str = 'mean'):
        assert reduction in ['mean', 'sum', 'batchmean']
        self.reduction = reduction
        
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = target * (target.log() - pred)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # batchmean
            return loss.sum() / pred.shape[0]

class CTCLoss(Loss):
    """Connectionist Temporal Classification Loss"""
    def __init__(self, blank: int = 0, reduction: str = 'mean'):
        self.blank = blank
        self.reduction = reduction
        
    def __call__(self, log_probs: Tensor, targets: Tensor, 
                 input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        # Forward-backward algorithm implementation
        # This is a simplified version and should be optimized for production use
        batch_size = log_probs.shape[0]
        losses = []
        
        for b in range(batch_size):
            T = input_lengths[b]
            S = target_lengths[b]
            
            # Initialize forward variables
            alpha = np.zeros((T, S * 2 + 1))
            alpha[0, 0] = log_probs[b, 0, self.blank]
            if S > 0:
                alpha[0, 1] = log_probs[b, 0, targets[b, 0]]
            
            # Forward pass
            for t in range(1, T):
                for s in range(S * 2 + 1):
                    if s % 2 == 0:  # blank
                        alpha[t, s] = alpha[t-1, s]
                        if s > 0:
                            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t-1, s-1])
                    else:  # label
                        alpha[t, s] = alpha[t-1, s]
                        if s > 1:
                            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t-1, s-1])
                        if s < S * 2 and targets[b, s//2] != targets[b, (s-2)//2]:
                            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t-1, s-2])
                            
                    alpha[t, s] += log_probs[b, t, targets[b, s//2] if s % 2 == 1 else self.blank]
            
            # Final loss
            loss = -np.logaddexp(alpha[T-1, S*2], alpha[T-1, S*2-1])
            losses.append(loss)
            
        losses = Tensor(np.array(losses))
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # none
            return losses
