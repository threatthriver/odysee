# Odysee: High-Performance Multi-Modal Deep Learning Framework

Odysee is a cutting-edge deep learning framework designed for efficient processing of both text and images, with support for context windows up to 4M tokens. Built with performance in mind, it leverages Rust and Metal acceleration on Apple Silicon.

## Key Features

- **4M Token Context Windows**: Efficiently handle extremely long sequences
- **Multi-Modal Processing**: Seamlessly work with both text and images
- **Metal Acceleration**: Optimized for Apple Silicon with Metal Performance Shaders
- **Memory Efficient**: Advanced gradient checkpointing and sparse attention
- **Cross-Language Integration**: Combines Python, Rust, and C++ for optimal performance

## Installation

```bash
# Clone the repository
git clone https://github.com/threatthriver/odysee.git
cd odysee

# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install -r requirements.txt

# Build Rust extensions
cd rust
cargo build --release
cd ..
```

## Quick Start

```python
from odysee.routing import MultiModalDynamicRouter, RoutingConfig
from PIL import Image

# Initialize router
config = RoutingConfig(
    routing_dim=1024,
    num_heads=8,
    max_context_length=4_000_000
)
router = MultiModalDynamicRouter(config)

# Process text
text_embeddings = get_embeddings(text)  # Your embedding function
routed_text, stats = router.route(text_embeddings)

# Process image
image = Image.open("example.jpg")
routed_image, stats = router.route(image, is_image=True)
```

## Architecture

Odysee uses a hybrid architecture combining:
- Python for high-level APIs and integration
- Rust for core processing and memory management
- Metal acceleration for GPU computations
- C++ for specialized numerical operations

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Odysee in your research, please cite:

```bibtex
@software{odysee2025,
  title = {Odysee: High-Performance Multi-Modal Deep Learning Framework},
  author = {ThreatThriver Team},
  year = {2025},
  url = {https://github.com/threatthriver/odysee}
}
```

## Acknowledgments

Special thanks to all contributors and the open source community.
