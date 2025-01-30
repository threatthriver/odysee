import numpy as np
from typing import Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
import logging
import sys
from enum import Enum, auto
from odysee_rust import MultiModalRouter

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
    text: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    audio: Optional[np.ndarray] = None
    video: Optional[np.ndarray] = None
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
                    if not isinstance(data, np.ndarray):
                        logger.error(f"{name.capitalize()} input must be a NumPy array")
                        return False
                    if data.dtype != np.float32:
                        logger.warning(f"Converting {name} input to float32")
                        setattr(self, name, data.astype(np.float32))
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
        strict_mode: bool = False
    ):
        """
        Initialize MultiModal Processor with configurable routing parameters
        
        Args:
            routing_dim (int): Dimension of routing embeddings
            num_heads (int): Number of attention heads
            max_seq_len (int): Maximum sequence length for processing
            strict_mode (bool): Enable strict input validation
        """
        try:
            self.router = MultiModalRouter(routing_dim, num_heads)
            self.routing_dim = routing_dim
            self.max_seq_len = max_seq_len
            self.strict_mode = strict_mode
        except Exception as e:
            logger.critical(f"Router initialization failed: {e}")
            raise
        
    def preprocess_input(self, input_data: MultimodalInput) -> MultimodalInput:
        """
        Preprocess and validate multimodal input
        
        Args:
            input_data (MultimodalInput): Raw input data
        
        Returns:
            MultimodalInput: Processed and validated input
        
        Raises:
            ValueError: If input is invalid in strict mode
        """
        if self.strict_mode and not input_data.validate():
            raise ValueError("Input validation failed in strict mode")
        
        def _validate_and_pad(data: Optional[np.ndarray], target_dim: int) -> Optional[np.ndarray]:
            if data is None:
                return None
            
            # Ensure float32 type
            data = data.astype(np.float32)
            
            # Flatten if needed
            if data.ndim > 2:
                data = data.reshape(-1, target_dim)
            
            # Pad or truncate
            if data.shape[0] > self.max_seq_len:
                logger.warning(f"Truncating input from {data.shape[0]} to {self.max_seq_len}")
                data = data[:self.max_seq_len]
            
            return data
        
        return MultimodalInput(
            text=_validate_and_pad(input_data.text, self.routing_dim),
            image=_validate_and_pad(input_data.image, self.routing_dim),
            audio=_validate_and_pad(input_data.audio, self.routing_dim),
            video=_validate_and_pad(input_data.video, self.routing_dim),
            metadata=input_data.metadata
        )
    
    def route_multimodal(self, input_data: MultimodalInput) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Route multimodal inputs through the MultiModalRouter
        
        Args:
            input_data (MultimodalInput): Preprocessed input data
        
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Routing results for each modality
        
        Raises:
            RuntimeError: If routing fails for any modality
        """
        results: Dict[str, Dict[str, np.ndarray]] = {}
        
        routing_methods = {
            'text': self.router.route_text,
            'image': self.router.route_text,  # Using text routing for image
            'audio': self.router.route_text,  # Placeholder for audio routing
        }
        
        for modality, data in [
            ('text', input_data.text),
            ('image', input_data.image),
            ('audio', input_data.audio)
        ]:
            if data is not None:
                try:
                    flattened_data = data.reshape(-1, self.routing_dim)
                    weights, indices = routing_methods[modality](
                        flattened_data.flatten().tolist(), 
                        batch_size=1, 
                        seq_len=flattened_data.shape[0]
                    )
                    results[modality] = {
                        'weights': np.array(weights),
                        'indices': np.array(indices)
                    }
                except Exception as e:
                    logger.error(f"Routing failed for {modality}: {e}")
                    if self.strict_mode:
                        raise RuntimeError(f"Routing failed for {modality}")
        
        return results

def main():
    # Example usage and demonstration
    processor = MultimodalProcessor(routing_dim=256, num_heads=4, strict_mode=True)
    
    # Generate sample multimodal data
    text_data = np.random.randn(128, 256).astype(np.float32)
    image_data = np.random.randn(224, 224, 256).astype(np.float32)
    
    input_data = MultimodalInput(
        text=text_data, 
        image=image_data, 
        metadata={'source': 'example_generation'}
    )
    
    try:
        processed_input = processor.preprocess_input(input_data)
        routing_results = processor.route_multimodal(processed_input)
        
        # Log routing results
        for modality, results in routing_results.items():
            logger.info(f"{modality.capitalize()} Routing Results:")
            logger.info(f"Weights shape: {results['weights'].shape}")
            logger.info(f"Indices shape: {results['indices'].shape}")
            logger.info(f"Sample weights: {results['weights'][:5]}")
            logger.info(f"Sample indices: {results['indices'][:5]}")
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
