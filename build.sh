#!/bin/bash

echo "==================================================="
echo "   Analisador de Imagens - Build Script (Unix/Linux)"
echo "==================================================="

echo
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.x using your package manager"
    exit 1
fi
python3 --version

echo
echo "[2/5] Installing system dependencies (tkinter)..."
if command -v apt-get &> /dev/null; then
    echo "Detected apt package manager (Ubuntu/Debian)"
    sudo apt-get update
    sudo apt-get install -y python3-tk python3-dev
elif command -v dnf &> /dev/null; then
    echo "Detected dnf package manager (Fedora)"
    sudo dnf install -y tkinter python3-devel
elif command -v yum &> /dev/null; then
    echo "Detected yum package manager (RHEL/CentOS)"
    sudo yum install -y tkinter python3-devel
elif command -v pacman &> /dev/null; then
    echo "Detected pacman package manager (Arch Linux)"
    sudo pacman -S --noconfirm tk python
else
    echo "WARNING: Could not detect package manager. Please install python3-tk manually:"
    echo "  Ubuntu/Debian: sudo apt-get install python3-tk"
    echo "  Fedora: sudo dnf install tkinter"
    echo "  Arch: sudo pacman -S tk"
fi

echo
echo "[3/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo
echo "[4/5] Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip

echo "Uninstalling existing PySimpleGUI..."
pip uninstall PySimpleGUI -y
pip cache purge

echo "Installing PySimpleGUI from private server..."
pip install --upgrade --extra-index-url https://PySimpleGUI.net/install PySimpleGUI

echo "Installing other dependencies..."
pip install opencv-python numpy scikit-learn Pillow

echo
echo "[5/5] Starting the application..."
cd src
python3 main.py

echo
echo "Application finished."