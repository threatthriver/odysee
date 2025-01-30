import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging
from odysee_rust import MultiModalRouter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MultimodalInput:
    """
    Structured data class for multimodal inputs
    Supports text, image, and audio inputs with flexible dimensions
    """
    text: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    audio: Optional[np.ndarray] = None

class MultimodalProcessor:
    def __init__(self, 
                 routing_dim: int = 256, 
                 num_heads: int = 4, 
                 max_seq_len: int = 512):
        """
        Initialize MultiModal Processor with configurable routing parameters
        
        Args:
            routing_dim (int): Dimension of routing embeddings
            num_heads (int): Number of attention heads
            max_seq_len (int): Maximum sequence length for processing
        """
        self.router = MultiModalRouter(routing_dim, num_heads)
        self.routing_dim = routing_dim
        self.max_seq_len = max_seq_len
        
    def preprocess_input(self, input_data: MultimodalInput) -> MultimodalInput:
        """
        Preprocess and validate multimodal input
        
        Args:
            input_data (MultimodalInput): Raw input data
        
        Returns:
            MultimodalInput: Processed and validated input
        """
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
            audio=_validate_and_pad(input_data.audio, self.routing_dim)
        )
    
    def route_multimodal(self, input_data: MultimodalInput) -> dict:
        """
        Route multimodal inputs through the MultiModalRouter
        
        Args:
            input_data (MultimodalInput): Preprocessed input data
        
        Returns:
            dict: Routing results for each modality
        """
        results = {}
        
        if input_data.text is not None:
            text_weights, text_indices = self.router.route_text(
                input_data.text.flatten().tolist(), 
                batch_size=1, 
                seq_len=input_data.text.shape[0]
            )
            results['text'] = {
                'weights': np.array(text_weights),
                'indices': np.array(text_indices)
            }
        
        if input_data.image is not None:
            # Flatten image to 2D for routing
            flattened_image = input_data.image.reshape(-1, self.routing_dim)
            image_weights, image_indices = self.router.route_text(
                flattened_image.flatten().tolist(), 
                batch_size=1, 
                seq_len=flattened_image.shape[0]
            )
            results['image'] = {
                'weights': np.array(image_weights),
                'indices': np.array(image_indices)
            }
        
        return results

def main():
    # Example usage and demonstration
    processor = MultimodalProcessor(routing_dim=256, num_heads=4)
    
    # Generate sample multimodal data
    text_data = np.random.randn(128, 256).astype(np.float32)
    image_data = np.random.randn(224, 224, 256).astype(np.float32)
    
    input_data = MultimodalInput(text=text_data, image=image_data)
    processed_input = processor.preprocess_input(input_data)
    
    routing_results = processor.route_multimodal(processed_input)
    
    # Log routing results
    for modality, results in routing_results.items():
        logger.info(f"{modality.capitalize()} Routing Results:")
        logger.info(f"Weights shape: {results['weights'].shape}")
        logger.info(f"Indices shape: {results['indices'].shape}")
        logger.info(f"Sample weights: {results['weights'][:5]}")
        logger.info(f"Sample indices: {results['indices'][:5]}")

if __name__ == "__main__":
    main()
