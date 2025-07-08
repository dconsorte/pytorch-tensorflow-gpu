#!/bin/bash

# Docker and Buildx Installation Script
# For Ubuntu 20.04+ systems
# Created by Dennis Consorte - https://dennisconsorte.com

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================${NC}"
echo -e "${BLUE}Docker & Buildx Installation Script${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

# Check if running with sudo
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please don't run this script with sudo${NC}"
   echo "The script will ask for sudo when needed"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check if Docker is already installed
if command_exists docker; then
    echo -e "${YELLOW}Docker is already installed${NC}"
    docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
    echo "Current version: $docker_version"
    
    read -p "Do you want to continue and check Buildx? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    echo -e "${BLUE}Installing Docker...${NC}"
    
    # Remove old versions
    echo "Removing old Docker versions (if any)..."
    sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    
    # Update package index
    echo "Updating package index..."
    sudo apt-get update
    
    # Install prerequisites
    echo "Installing prerequisites..."
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    echo "Adding Docker GPG key..."
    sudo mkdir -m 0755 -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Set up repository
    echo "Setting up Docker repository..."
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Update package index again
    sudo apt-get update
    
    # Install Docker Engine
    echo "Installing Docker Engine..."
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Add current user to docker group
    echo "Adding $USER to docker group..."
    sudo usermod -aG docker $USER
    
    echo -e "${GREEN}Docker installed successfully!${NC}"
fi

# 2. Check Docker Buildx
echo ""
echo -e "${BLUE}Checking Docker Buildx...${NC}"

if docker buildx version >/dev/null 2>&1; then
    echo -e "${GREEN}Docker Buildx is already installed${NC}"
    buildx_version=$(docker buildx version | awk '{print $2}')
    echo "Version: $buildx_version"
else
    echo -e "${YELLOW}Installing Docker Buildx plugin...${NC}"
    
    # Install buildx plugin
    sudo apt-get update
    sudo apt-get install -y docker-buildx-plugin
    
    # Create buildx builder
    docker buildx create --name mybuilder --driver docker-container --use
    docker buildx inspect --bootstrap
    
    echo -e "${GREEN}Docker Buildx installed successfully!${NC}"
fi

# 3. Install NVIDIA Container Toolkit (for GPU support)
echo ""
echo -e "${BLUE}Checking NVIDIA Container Toolkit...${NC}"

if command_exists nvidia-container-toolkit; then
    echo -e "${GREEN}NVIDIA Container Toolkit is already installed${NC}"
else
    echo -e "${YELLOW}Installing NVIDIA Container Toolkit...${NC}"
    
    # Check if NVIDIA GPU exists
    if ! command_exists nvidia-smi; then
        echo -e "${YELLOW}Warning: nvidia-smi not found. You may need to install NVIDIA drivers first.${NC}"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
    
    # Configure production repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Update and install
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo -e "${GREEN}NVIDIA Container Toolkit installed successfully!${NC}"
fi

# 4. Verify installation
echo ""
echo -e "${BLUE}Verifying installation...${NC}"

# Check Docker
if docker run hello-world >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker is working correctly${NC}"
else
    echo -e "${RED}✗ Docker test failed${NC}"
fi

# Check Buildx
if docker buildx version >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker Buildx is available${NC}"
else
    echo -e "${RED}✗ Docker Buildx not found${NC}"
fi

# Check GPU support (if NVIDIA GPU exists)
if command_exists nvidia-smi; then
    if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}✓ GPU support is working${NC}"
    else
        echo -e "${YELLOW}⚠ GPU support test failed (may need reboot)${NC}"
    fi
fi

# 5. Final instructions
echo ""
echo -e "${BLUE}===================================${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "1. You need to log out and back in for group changes to take effect"
echo "   Or run: newgrp docker"
echo ""
echo "2. To test Docker without sudo:"
echo "   docker run hello-world"
echo ""
echo "3. To test GPU support (after logout/login):"
echo "   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi"
echo ""
echo -e "${BLUE}===================================${NC}"

# Ask if user wants to activate docker group now
read -p "Activate docker group now? (recommended) (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Activating docker group..."
    newgrp docker
fi