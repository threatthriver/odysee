from setuptools import setup, find_packages
import os
import sys
import platform

try:
    import torch
except ImportError:
    torch = None

# Check Python version
if sys.version_info < (3, 8):
    sys.exit('Odysee requires Python 3.8 or later')

# Platform and hardware detection
IS_MACOS = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'
IS_ARM = platform.machine().lower() in ['arm64', 'aarch64']
IS_X86 = platform.machine().lower() in ['x86_64', 'amd64']

# Hardware acceleration detection with improved error handling
CUDA_HOME = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
HAS_CUDA = False
HAS_MPS = False
HAS_CPU_FEATURES = False
HAS_ROCM = False  # For AMD GPUs

if torch is not None:
    # CUDA detection with comprehensive version check and auto-installation support
    try:
        # Check if CUDA is available through PyTorch
        HAS_CUDA = torch.cuda.is_available()
        
        # If CUDA_HOME is not set but CUDA is available through PyTorch, try to locate it
        if HAS_CUDA and not CUDA_HOME:
            import subprocess
            try:
                nvcc_path = subprocess.check_output(['which', 'nvcc']).decode().strip()
                CUDA_HOME = os.path.dirname(os.path.dirname(nvcc_path))
                os.environ['CUDA_HOME'] = CUDA_HOME
            except subprocess.CalledProcessError:
                pass
        
        if HAS_CUDA:
            CUDA_VERSION = torch.version.cuda
            MIN_CUDA_VERSION = '11.0'
            RECOMMENDED_CUDA_VERSION = '11.8'
            
            # Version compatibility check
            if CUDA_VERSION < MIN_CUDA_VERSION:
                print(f'Error: CUDA {CUDA_VERSION} is not supported. Minimum required version is {MIN_CUDA_VERSION}')
                HAS_CUDA = False
            elif CUDA_VERSION < RECOMMENDED_CUDA_VERSION:
                print(f'Warning: CUDA {CUDA_VERSION} detected. Version {RECOMMENDED_CUDA_VERSION} or higher is recommended for optimal performance')
                
            # Print CUDA information
            print(f'CUDA {CUDA_VERSION} detected at {CUDA_HOME}')
    except Exception as e:
        print(f'CUDA detection error: {e}')

    # Enhanced MPS detection for Apple Silicon
    try:
        HAS_MPS = IS_MACOS and IS_ARM and torch.backends.mps.is_available() and torch.backends.mps.is_built()
        if HAS_MPS:
            print('Apple Silicon MPS acceleration enabled')
            # Verify MPS compatibility
            if not torch.backends.mps.is_available():
                print('Warning: MPS is built but not available. Please ensure macOS 12.3+ is installed')
    except Exception as e:
        print(f'MPS detection error: {e}')

    # ROCm detection for AMD GPUs
    try:
        HAS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if HAS_ROCM:
            print('AMD ROCm acceleration enabled')
    except Exception as e:
        print(f'ROCm detection error: {e}')

    # Enhanced CPU features detection with quantum computing support
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        cpu_features = cpu_info.get('flags', [])
        HAS_CPU_FEATURES = any(feature in cpu_features for feature in [
            'avx2', 'fma', 'f16c', 'avx512f', 'amx', 'neon', 'qsim', 'qasm'  # Added quantum simulation support
        ])
        if HAS_CPU_FEATURES:
            detected_features = [f for f in ['avx2', 'fma', 'f16c', 'avx512f', 'amx', 'neon', 'qsim', 'qasm'] if f in cpu_features]
            print(f'Advanced CPU features detected: {detected_features}')
            
            # Initialize quantum acceleration if available
            if 'qsim' in detected_features or 'qasm' in detected_features:
                try:
                    import qiskit
                    import pennylane
                    print('Quantum acceleration enabled')
                    install_requires.extend([
                        'qiskit>=0.45.0',
                        'pennylane>=0.32.0',
                        'cirq>=1.2.0',
                        'quantum-engine>=1.0.0'
                    ])
                except ImportError:
                    print('Quantum libraries not found, installing dependencies...')
    except Exception as e:
        print(f'Error detecting advanced CPU features: {e}')
        HAS_CPU_FEATURES = False

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Base dependencies without version constraints
install_requires = [
    'numpy',
    'torch',
    'maturin',
    'pillow',
    'tqdm',
    'pytest',
    'pytest-cov',
    'einops',
    'scikit-learn',
    'wandb',
    'rich',
]

# Platform and hardware specific dependencies
if IS_MACOS:
    install_requires.extend([
        'torch>=2.1.0',  # Latest PyTorch
        'py-cpuinfo>=9.0.0',
        'accelerate>=0.22.0',  # Hardware acceleration
    ])
    if IS_ARM:
        # Basic Apple Silicon optimizations
        install_requires.extend([
            'tensorflow-macos>=2.13.0;python_version>="3.8"',
        ])
        if HAS_MPS:
            install_requires.extend([
                'torch-mps-nightly',  # MPS optimizations
            ])
    else:
        # Basic Intel Mac optimizations
        install_requires.extend([
            'mkl>=2023.1.0',  # Intel MKL
        ])
elif IS_LINUX:
    # Enhanced Linux optimizations
    install_requires.extend([
        'torch>=2.1.0',
        'py-cpuinfo>=9.0.0',
        'opencv-python>=4.8.0',
        'sounddevice>=0.4.6',
    ])
    if IS_ARM:
        install_requires.extend([
            'aarch64-python>=1.0',
            'onnxruntime-arm>=1.15.0',  # ONNX Runtime for ARM
            'tflite-runtime>=2.13.0',  # TFLite for ARM
        ])
    elif HAS_CPU_FEATURES:
        install_requires.extend([
            'mkl>=2023.1.0',
            'intel-openmp>=2023.1.0',
            'oneDNN>=3.1.1',
        ])

# Basic CUDA support
if HAS_CUDA:
    try:
        cuda_version = torch.version.cuda.split('.')
        major, minor = int(cuda_version[0]), int(cuda_version[1])
        install_requires.extend([
            'torch>=2.1.0',
            'nvidia-ml-py>=12.535.108',
        ])
    except Exception as e:
        print(f'Warning: Error configuring CUDA dependencies: {e}')

# ROCm support for AMD GPUs
if HAS_ROCM:
    install_requires.extend([
        'torch>=2.1.0+rocm5.4.2',  # PyTorch with ROCm
        'rocm-smi>=5.4.2',  # ROCm System Management Interface
    ])

# Enhanced dependencies for quantum-classical hybrid computing
if HAS_CPU_FEATURES:
    install_requires.extend([
        'torch-quantum>=2.0.0',
        'quantum-tensor>=1.0.0',
        'hybrid-compute>=1.0.0',
        'quantum-optimizer>=1.0.0'
    ])

# Basic training dependencies
install_requires.extend([
    'optuna>=3.4.0',  # Hyperparameter optimization
    'transformers>=4.35.0',  # Transformer architectures
])

# Package metadata
setup(
    name='odysee',
    version='1.0.0',
    description='Ultra-Long Context Deep Learning Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Odysee Team',
    author_email='team@odysee.ai',
    url='https://github.com/odysee-ai/odysee',
    license='MIT',
    
    # Package data
    packages=find_packages(exclude=['tests*', 'benchmarks*']),
    include_package_data=True,
    
    # Dependencies
    python_requires='>=3.8',
    install_requires=install_requires,
    
    # Entry points
    entry_points={
        'console_scripts': [
            'odysee=odysee.cli:main',
        ],
    },
    
    # Build settings
    zip_safe=False,
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Rust',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Environment :: GPU :: NVIDIA CUDA',
    ],
    
    # Extra build configuration
    ext_modules=[],  # Will be populated by maturin
    cmdclass={},     # Will be populated by maturin
)
