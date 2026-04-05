#!/usr/bin/env bash
set -e

# Ensure we are running from the third_party directory
if [ "$(basename "$PWD")" != "third_party" ]; then
    echo "Error: Please run this script from the third_party directory."
    exit 1
fi

# Clone the CoTracker repository if it doesn't already exist
if [ -d "co-tracker" ]; then
    echo "CoTracker repository already exists. Skipping clone."
else
    git clone git@github.com:GeoRAMIEL/co-tracker.git
fi
cd co-tracker

# make sure we are using the venv in the root of the project
echo "This script installs python dependencies for CoTracker. Make sure you have activated the virtual environment in the root of the project before running this script."
read -p "Press enter to continue or Ctrl+C to cancel..."
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard

mkdir -p checkpoints
cd checkpoints
# download the offline (single window) model if it doesn't already exist
if [ ! -f "scaled_offline.pth" ]; then
    echo "Downloading CoTracker3 scaled_offline.pth checkpoint..."
    wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
fi
# download the online (sliding window) model if it doesn't already exist
if [ ! -f "scaled_online.pth" ]; then
    echo "Downloading CoTracker3 scaled_online.pth checkpoint..."
    wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
fi
echo "CoTracker3 checkpoints are ready."
echo "CoTracker installation complete."
cd ../..
