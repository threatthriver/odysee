import numpy as np
import torch
from typing import Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
import logging
import sys
from enum import Enum, auto
from odysee_rust import MultiModalRouter
from odysee.routing import DynamicRouter, RoutingConfig
from odysee.cuda_ops import quantum_phase_encoding, flash_attention

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('multimodal_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Enum for different modality types"""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()

@dataclass
class MultimodalInput:
    """
    Structured data class for multimodal inputs
    Supports flexible input types with validation
    """
    text: Optional[Union[np.ndarray, torch.Tensor]] = None
    image: Optional[Union[np.ndarray, torch.Tensor]] = None
    audio: Optional[Union[np.ndarray, torch.Tensor]] = None
    video: Optional[Union[np.ndarray, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """
        Validate input data types and dimensions
        
        Returns:
            bool: Whether input is valid
        """
        try:
            for name, data in [
                ('text', self.text),
                ('image', self.image),
                ('audio', self.audio),
                ('video', self.video)
            ]:
                if data is not None:
                    if not isinstance(data, (np.ndarray, torch.Tensor)):
                        logger.error(f"{name.capitalize()} input must be a NumPy array or PyTorch tensor")
                        return False
                    if isinstance(data, np.ndarray) and data.dtype != np.float32:
                        logger.warning(f"Converting {name} input to float32")
                        setattr(self, name, data.astype(np.float32))
                    elif isinstance(data, torch.Tensor) and data.dtype != torch.float32:
                        logger.warning(f"Converting {name} input to float32")
                        setattr(self, name, data.to(torch.float32))
            return True
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False

class MultimodalProcessor:
    def __init__(
        self, 
        routing_dim: int = 256, 
        num_heads: int = 4, 
        max_seq_len: int = 512,
        strict_mode: bool = False,
        use_cuda: bool = True
    ):
        """
        Initialize MultiModal Processor with configurable routing parameters
        
        Args:
            routing_dim: Dimension of routing embeddings
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for processing
            strict_mode: Enable strict input validation
            use_cuda: Use CUDA operations when available
        """
        try:
            config = RoutingConfig(
                routing_dim=routing_dim,
                num_heads=num_heads,
                max_context_length=max_seq_len,
                use_flash_attention=use_cuda
            )
            self.router = DynamicRouter(config)
            self.routing_dim = routing_dim
            self.max_seq_len = max_seq_len
            self.strict_mode = strict_mode
            self.use_cuda = use_cuda and torch.cuda.is_available()
            
            if self.use_cuda:
                logger.info("Using CUDA operations")
            else:
                logger.info("Using CPU operations")
                
        except Exception as e:
            logger.critical(f"Router initialization failed: {e}")
            raise
        
    def preprocess_input(self, input_data: MultimodalInput) -> MultimodalInput:
        """
        Preprocess and validate multimodal input
        
        Args:
            input_data: Raw input data
        
        Returns:
            MultimodalInput: Processed and validated input
        
        Raises:
            ValueError: If input is invalid in strict mode
        """
        if self.strict_mode and not input_data.validate():
            raise ValueError("Input validation failed in strict mode")
        
        def _validate_and_pad(data: Optional[Union[np.ndarray, torch.Tensor]], target_dim: int) -> Optional[torch.Tensor]:
            if data is None:
                return None
            
            # Convert to PyTorch tensor
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            
            # Ensure float32 type
            data = data.to(torch.float32)
            
            # Move to GPU if available
            if self.use_cuda:
                data = data.cuda()
            
            # Handle different input shapes
            if len(data.shape) == 2:  # (seq_len, dim)
                data = data.unsqueeze(0)  # Add batch dimension
            elif len(data.shape) == 3 and data.shape[-1] != target_dim:  # (batch, seq_len, wrong_dim)
                raise ValueError(f"Expected dimension {target_dim}, got {data.shape[-1]}")
            elif len(data.shape) > 3:  # For image/video data
                orig_shape = data.shape
                data = data.view(-1, target_dim)  # Flatten spatial dimensions
                data = data.unsqueeze(0)  # Add batch dimension
            
            # Pad or truncate sequence length
            if data.shape[1] > self.max_seq_len:
                logger.warning(f"Truncating input from {data.shape[1]} to {self.max_seq_len}")
                data = data[:, :self.max_seq_len, :]
            
            return data
        
        return MultimodalInput(
            text=_validate_and_pad(input_data.text, self.routing_dim),
            image=_validate_and_pad(input_data.image, self.routing_dim),
            audio=_validate_and_pad(input_data.audio, self.routing_dim),
            video=_validate_and_pad(input_data.video, self.routing_dim),
            metadata=input_data.metadata
        )
    
    def route_multimodal(self, input_data: MultimodalInput) -> Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]]:
        """
        Route multimodal inputs through the MultiModalRouter
        
        Args:
            input_data: Preprocessed input data
        
        Returns:
            Dict containing routing results for each modality
        
        Raises:
            RuntimeError: If routing fails for any modality
        """
        results = {}
        
        for modality, data in [
            ('text', input_data.text),
            ('image', input_data.image),
            ('audio', input_data.audio),
            ('video', input_data.video)
        ]:
            if data is not None:
                try:
                    # Route through dynamic router
                    weights, indices = self.router.route(
                        data,
                        is_image=(modality in ['image', 'video'])
                    )
                    
                    # Convert results to appropriate format
                    if isinstance(weights, np.ndarray):
                        weights = torch.from_numpy(weights)
                    if isinstance(indices, np.ndarray):
                        indices = torch.from_numpy(indices)
                    
                    if self.use_cuda:
                        weights = weights.cuda()
                        indices = indices.cuda()
                    
                    results[modality] = {
                        'weights': weights,
                        'indices': indices
                    }
                except Exception as e:
                    logger.error(f"Routing failed for {modality}: {e}")
                    if self.strict_mode:
                        raise RuntimeError(f"Routing failed for {modality}")
        
        return results

def main():
    # Example usage and demonstration
    processor = MultimodalProcessor(
        routing_dim=256,
        num_heads=4,
        strict_mode=True,
        use_cuda=True
    )
    
    # Generate sample multimodal data
    batch_size = 1
    seq_len = 128
    text_data = torch.randn(batch_size, seq_len, 256, dtype=torch.float32)  # (batch, seq, dim)
    
    # Image data with spatial dimensions
    height, width = 224, 224
    image_data = torch.randn(batch_size, height, width, 256, dtype=torch.float32)  # (batch, h, w, dim)
    
    input_data = MultimodalInput(
        text=text_data,
        image=image_data,
        metadata={'source': 'example_generation'}
    )
    
    try:
        # Preprocess the input
        processed_input = processor.preprocess_input(input_data)
        
        # Route through the model
        routing_results = processor.route_multimodal(processed_input)
        
        # Log routing results
        for modality, results in routing_results.items():
            logger.info(f"\n{modality.capitalize()} Routing Results:")
            logger.info(f"Weights shape: {results['weights'].shape}")
            logger.info(f"Indices shape: {results['indices'].shape}")
            logger.info(f"Sample weights: {results['weights'][:5]}")
            logger.info(f"Sample indices: {results['indices'][:5]}")
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
