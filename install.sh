#!/bin/bash

# Update package lists (may not be necessary on all systems)
sudo apt update

# Install prerequisites (adjust for your distribution if needed)
sudo apt install python3-pip

# Install pipx
python3 -m pip install pipx

# Install poetry using pipx
pipx install poetry

# Print confirmation message
echo "pipx and poetry are now installed!"