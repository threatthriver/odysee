from odysee.routing import MultiModalRouter, RoutingConfig
from PIL import Image
import torch
import numpy as np

def main():
    # Initialize the router with a configuration
    config = RoutingConfig(
        routing_dim=1024,
        num_heads=8,
        max_context_length=4_000_000
    )
    router = MultiModalRouter(routing_dim=config.routing_dim, num_heads=config.num_heads)
    
    # Example text processing
    text = "This is a sample text that will be processed by Odysee."
    # Simulate text embeddings (batch_size=1, seq_len=11, routing_dim=1024)
    text_embeddings = np.random.randn(1 * 11, 1024).astype(np.float32)
    weights, indices = router.route_text(text_embeddings, batch_size=1, seq_len=11)
    print("Text processing complete!")
    print(f"Routing weights shape: {weights.shape}")
    print(f"Routing indices shape: {indices.shape}")
    
    # Example image processing (commented out as we don't have a sample image)
    # image_embeddings = np.random.randn(32, 32, 1024).astype(np.float32)  # 32x32 image patches
    # weights, indices = router.route_image(image_embeddings, image_size=(32, 32))
    # print(f"Image processing complete!")
    # print(f"Routing weights shape: {weights.shape}")
    # print(f"Routing indices shape: {indices.shape}")

if __name__ == "__main__":
    main()
