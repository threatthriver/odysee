from setuptools import setup, find_packages
import os
import sys
import platform

# Check Python version
if sys.version_info < (3, 8):
    sys.exit('Odysee requires Python 3.8 or later')

# Platform specific settings
IS_MACOS = platform.system() == 'Darwin'
IS_ARM = platform.machine() == 'arm64'
CUDA_HOME = os.environ.get('CUDA_HOME')
HAS_CUDA = CUDA_HOME is not None

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Base dependencies
install_requires = [
    'numpy>=1.20.0',
    'torch>=1.9.0',
    'maturin>=1.0.0',
    'pillow>=8.0.0',
    'tqdm>=4.62.0',
    'pytest>=6.0.0',
    'pytest-cov>=2.0.0',
    'einops>=0.6.0',
    'scikit-learn>=1.0.0',
    'wandb>=0.15.0',
    'rich>=13.0.0',
]

# Platform specific dependencies
if IS_MACOS:
    if IS_ARM:
        # M1/M2 Mac specific optimizations
        install_requires.extend([
            'torch>=2.0.0',  # Better M1 support
            'accelerate>=0.20.0',  # Hardware acceleration
        ])
    else:
        # Intel Mac
        install_requires.extend([
            'torch>=1.9.0',
            'mkl>=2021',  # Intel MKL optimization
        ])

# CUDA support
if HAS_CUDA:
    install_requires.extend([
        'cupy-cuda11x>=12.0.0',
        'torch>=2.0.0+cu118',  # CUDA 11.8 support
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
