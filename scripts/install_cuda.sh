#!/bin/bash

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -sr)

# Map newer Ubuntu versions to the latest supported version
if [ "${UBUNTU_VERSION}" = "24.04" ]; then
    CUDA_REPO_VERSION="2204"
    echo "Ubuntu 24.04 detected, using Ubuntu 22.04 CUDA repository"
else
    CUDA_REPO_VERSION=${UBUNTU_VERSION//./}
fi

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${CUDA_REPO_VERSION}/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-0

# Set environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Verify installation
nvcc --version