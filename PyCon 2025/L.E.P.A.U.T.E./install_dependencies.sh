#!/bin/bash

# Script Name: install_dependencies.sh
# Purpose: Create a Python virtual environment, install pipenv 3.13, and install specified Python dependencies

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Starting installation of Python dependencies in a virtual environment..."
echo "----------------------------------------"

# 1. Check if Python 3.8 or higher is installed
echo -n "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>/dev/null | cut -d' ' -f2)
if [[ -z "$PYTHON_VERSION" ]]; then
    echo -e "${RED}Python not detected! Please install Python 3.8 or higher.${NC}"
    exit 1
fi
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 8) )); then
    echo -e "${RED}Python version $PYTHON_VERSION is too low, requires >= 3.8${NC}"
    exit 1
else
    echo -e "${GREEN}Python version $PYTHON_VERSION detected${NC}"
fi

# 2. Create a virtual environment
VENV_DIR="venv_$(date +%Y%m%d_%H%M%S)"
echo -n "Creating virtual environment at $VENV_DIR..."
if python3 -m venv "$VENV_DIR"; then
    echo -e "${GREEN}Virtual environment created successfully${NC}"
else
    echo -e "${RED}Failed to create virtual environment${NC}"
    exit 1
fi

# 3. Activate the virtual environment
echo -n "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}Virtual environment activated${NC}"
else
    echo -e "${RED}Failed to activate virtual environment${NC}"
    exit 1
fi

# 4. Upgrade pip in the virtual environment
echo -n "Upgrading pip..."
if pip install --upgrade pip >/dev/null 2>&1; then
    echo -e "${GREEN}pip upgraded successfully${NC}"
else
    echo -e "${RED}Failed to upgrade pip${NC}"
    deactivate
    exit 1
fi

# 5. Install pipenv version 3.13
echo -n "Installing pipenv==2023.10.24..."
if pip install pipenv==2023.10.24 >/dev/null 2>&1; then
    echo -e "${GREEN}pipenv 2023.10.24 installed successfully${NC}"
else
    echo -e "${RED}Failed to install pipenv${NC}"
    deactivate
    exit 1
fi

# 6. Create a Pipfile with the required dependencies
echo -n "Creating Pipfile with dependencies..."
cat > Pipfile << EOL
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
torch = ">=2.3.0"
kornia = ">=0.7.2"
opencv-python = ">=4.8.0"
numpy = ">=1.24.0"
torchvision = ">=0.15.0"
pytorch-metric-learning = ">=2.0.0"
scipy = ">=1.10.0"
matplotlib = ">=3.8.0"

[requires]
python_version = "3.8"
EOL

if [[ -f "Pipfile" ]]; then
    echo -e "${GREEN}Pipfile created successfully${NC}"
else
    echo -e "${RED}Failed to create Pipfile${NC}"
    deactivate
    exit 1
fi

# 7. Install dependencies using pipenv
echo "Installing dependencies with pipenv..."
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

for DEP in "${DEPENDENCIES[@]}"; do
    PKG=$(echo $DEP | cut -d'>=' -f1)
    echo -n "Installing $PKG..."
    if pipenv install "$DEP" >/dev/null 2>&1; then
        INSTALLED_VERSION=$(pipenv run pip show $PKG 2>/dev/null | grep Version | awk '{print $2}')
        echo -e "${GREEN}$PKG installed, version: $INSTALLED_VERSION${NC}"
    else
        echo -e "${RED}Failed to install $PKG${NC}"
        deactivate
        exit 1
    fi
done

# 8. Verify installations
echo "----------------------------------------"
echo "Verifying installed dependencies..."
ALL_INSTALLED=1
for DEP in "${DEPENDENCIES[@]}"; do
    PKG=$(echo $DEP | cut -d'>=' -f1)
    INSTALLED_VERSION=$(pipenv run pip show $PKG 2>/dev/null | grep Version | awk '{print $2}')
    if [[ -z "$INSTALLED_VERSION" ]]; then
        echo -e "${RED}$PKG not installed${NC}"
        ALL_INSTALLED=0
    else
        echo -e "${GREEN}$PKG verified, version: $INSTALLED_VERSION${NC}"
    fi
done

# 9. Final message
echo "----------------------------------------"
if [[ $ALL_INSTALLED -eq 1 ]]; then
    echo -e "${GREEN}All dependencies installed successfully in $VENV_DIR!${NC}"
    echo "To activate the virtual environment, run:"
    echo "  source $VENV_DIR/bin/activate"
    echo "To use pipenv, run commands like 'pipenv run python' from the directory containing the Pipfile."
else
    echo -e "${RED}Some dependencies failed to install. Please check the errors above.${NC}"
    deactivate
    exit 1
fi

# Deactivate the virtual environment
deactivate
echo -e "${GREEN}Installation process complete!${NC}"