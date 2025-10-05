#!/bin/bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
echo "Starting system requirements check..."
echo "Checking: GPU (MPS/CUDA), RAM (>=8GB), VS Code, CPU high-load capability, Webcam"
echo "----------------------------------------"
# 0 = pass, 1 = fail
CHECK_FAILED=0
echo -n "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>/dev/null | cut -d' ' -f2)
if [[ -z "$PYTHON_VERSION" ]]; then
    echo -e "${RED}Python not detected! Please install Python 3.8 or higher.${NC}"
    CHECK_FAILED=1
else
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 8) )); then
        echo -e "${RED}Python version $PYTHON_VERSION is too low, requires >= 3.8${NC}"
        CHECK_FAILED=1
    else
        echo -e "${GREEN}Python version $PYTHON_VERSION meets requirements${NC}"
    fi
fi
echo -n "Checking GPU support (CUDA/MPS)..."
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
MPS_AVAILABLE=$(python3 -c "import torch; print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())" 2>/dev/null)
if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
    echo -e "${GREEN}CUDA GPU detected, driver version: $CUDA_VERSION${NC}"
elif [[ "$MPS_AVAILABLE" == "True" ]]; then
    echo -e "${GREEN}MPS (Apple Silicon GPU) support detected${NC}"
else
    echo -e "${YELLOW}No CUDA or MPS GPU detected, may require CPU computing or GPU driver installation${NC}"
fi
echo -n "Checking system RAM..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    TOTAL_RAM=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2}' | tr -d ' GB')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    TOTAL_RAM=$(free -m | awk '/Mem:/ {print $2}' | awk '{print int($1/1024)}')
else
    echo -e "${RED}Unsupported operating system, cannot check RAM${NC}"
    CHECK_FAILED=1
fi
if [[ -z "$TOTAL_RAM" || "$TOTAL_RAM" -lt 8 ]]; then
    echo -e "${RED}RAM $TOTAL_RAM GB is insufficient, requires >= 8GB${NC}"
    CHECK_FAILED=1
else
    echo -e "${GREEN}RAM $TOTAL_RAM GB meets requirements${NC}"
fi
echo -n "Checking VS Code installation..."
if command -v code >/dev/null 2>&1; then
    VSCODE_VERSION=$(code --version | head -n 1)
    echo -e "${GREEN}VS Code installed, version: $VSCODE_VERSION${NC}"
else
    echo -e "${YELLOW}VS Code not detected, recommended for development${NC}"
fi
echo -n "Checking CPU performance..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    CPU_CORES=$(sysctl -n hw.ncpu)
    CPU_MODEL=$(system_profiler SPHardwareDataType | grep "Processor Name" | awk -F': ' '{print $2}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CPU_CORES=$(nproc)
    CPU_MODEL=$(cat /proc/cpuinfo | grep "model name" | head -n 1 | awk -F': ' '{print $2}')
fi
if [[ "$CPU_CORES" -ge 4 ]]; then
    echo -e "${GREEN}CPU model: $CPU_MODEL, cores: $CPU_CORES, suitable for high-load work${NC}"
else
    echo -e "${YELLOW}CPU cores: $CPU_CORES, may not be suitable for high-load work, recommend >= 4 cores${NC}"
fi
echo -n "Checking webcam functionality..."
WEBCAM_CHECK=$(python3 -c "import cv2; cap = cv2.VideoCapture(0); success, _ = cap.read() if cap.isOpened() else (False, None); cap.release(); print(success)" 2>/dev/null)
if [[ "$WEBCAM_CHECK" == "True" ]]; then
    echo -e "${GREEN}Webcam detected and functional${NC}"
else
    echo -e "${YELLOW}No functional webcam detected, please ensure a webcam is connected and accessible${NC}"
fi
echo "Checking Python dependencies..."
DEPENDENCIES=(
    "torch>=2.3.0"
    "kornia>=0.7.2"
    "opencv-python>=4.8.0"
    "numpy>=1.24.0"
    "torchvision>=0.15.0"
    "pytorch-metric-learning>=2.0.0"
    "scipy>=1.10.0"
    "matplotlib>=3.8.0"
)
DEPENDENCY_INSTALLED=0
for DEP in "${DEPENDENCIES[@]}"; do
    PKG=$(echo $DEP | cut -d'>=' -f1)
    REQ_VERSION=$(echo $DEP | cut -d'>=' -f2)
    echo -n "Checking $PKG..."
    INSTALLED_VERSION=$(pip show $PKG 2>/dev/null | grep Version | awk '{print $2}')
    if [[ -z "$INSTALLED_VERSION" ]]; then
        echo -e "${RED}$PKG not installed, please install with 'pip install $DEP'${NC}"
    else
        echo -e "${GREEN}$PKG installed, version: $INSTALLED_VERSION${NC}"
        DEPENDENCY_INSTALLED=1
    fi
done
echo "----------------------------------------"
echo "System check complete!"
if [[ $CHECK_FAILED -eq 0 && $DEPENDENCY_INSTALLED -eq 1 ]]; then
    echo -e "${GREEN}Result: PASS${NC}"
    echo "All critical requirements (Python, RAM, and at least one dependency) are met."
else
    echo -e "${RED}Result: NOT PASS${NC}"
    echo "One or more critical requirements (Python >= 3.8, RAM >= 8GB, or dependencies) are not met."
fi
echo "Please address any issues indicated above."