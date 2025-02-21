#!/bin/bash
# setup_env.sh - Script to set up the Python virtual environment for golLab.py

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check if the script is run with sudo/root privileges (optional, for system-wide packages)
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Warning: Running as root. This is not necessary and may cause issues. Run without sudo if possible.${NC}"
fi

# Check if virtualenv is installed, install if not
if ! command -v virtualenv &> /dev/null; then
    echo -e "${RED}virtualenv not found. Installing...${NC}"
    sudo apt update
    sudo apt install python3-virtualenv -y
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install virtualenv. Please check your system permissions or internet connection.${NC}"
        exit 1
    fi
    echo -e "${GREEN}virtualenv installed successfully.${NC}"
else
    echo -e "${GREEN}virtualenv is already installed.${NC}"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "gol_env" ]; then
    echo -e "${GREEN}Creating virtual environment 'gol_env'...${NC}"
    virtualenv gol_env -p python3
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to create virtual environment. Please check Python3 installation.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment 'gol_env' created successfully.${NC}"
else
    echo -e "${GREEN}Virtual environment 'gol_env' already exists.${NC}"
fi

# Activate virtual environment and install dependencies
echo -e "${GREEN}Activating virtual environment and installing dependencies...${NC}"
source gol_env/bin/activate

if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to upgrade pip. Please check your internet connection or permissions.${NC}"
        deactivate
        exit 1
    fi

    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install dependencies from requirements.txt. Check requirements.txt and internet connection.${NC}"
        deactivate
        exit 1
    fi
    echo -e "${GREEN}Dependencies installed from requirements.txt successfully.${NC}"
else
    echo -e "${RED}Error: requirements.txt not found. Please ensure it exists in the project directory.${NC}"
    deactivate
    exit 1
fi

echo -e "${GREEN}Environment setup complete. Activate with: source gol_env/bin/activate${NC}"
echo -e "${GREEN}Run the application with: python golLab.py${NC}"
deactivate
