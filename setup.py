from setuptools import setup, find_packages
import os
import sys

# Check Python version
if sys.version_info < (3, 8):
    sys.exit('Odysee requires Python 3.8 or later')

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Package metadata
setup(
    name='odysee',
    version='1.0.0',
    description='Ultra-Long Context Deep Learning Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/odysee',
    license='MIT',
    
    # Package data
    packages=find_packages(exclude=['tests*', 'benchmarks*', 'examples*']),
    include_package_data=True,
    zip_safe=False,
    
    # Dependencies
    python_requires='>=3.8',
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'cuda': [
            'cupy-cuda11x>=12.0.0',
            'torch>=2.0.0',
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'hypothesis>=6.0.0',
        ],
        'benchmark': [
            'psutil>=5.0.0',
            'pandas>=2.0.0',
            'matplotlib>=3.0.0',
        ],
        'full': [
            'cupy-cuda11x>=12.0.0',
            'torch>=2.0.0',
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'hypothesis>=6.0.0',
            'psutil>=5.0.0',
            'pandas>=2.0.0',
            'matplotlib>=3.0.0',
        ],
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'odysee-benchmark=odysee.cli.benchmark:main',
            'odysee-train=odysee.cli.train:main',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Rust',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent',
        'Environment :: GPU :: NVIDIA CUDA',
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://odysee.readthedocs.io',
        'Source': 'https://github.com/yourusername/odysee',
        'Tracker': 'https://github.com/yourusername/odysee/issues',
    },
    
    # Build settings
    rust_extensions=[
        'odysee.core.tensor_ops',
        'odysee.core.attention_ops',
    ],
)
