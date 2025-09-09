#!/bin/bash

echo "==================================================="
echo "   Analisador de Imagens - Build Script (Unix/Linux)"
echo "==================================================="

echo
echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.x using your package manager"
    exit 1
fi
python3 --version

echo
echo "[2/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo
echo "[3/4] Activating virtual environment and installing dependencies..."
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
echo "[4/4] Starting the application..."
cd src
python main.py

echo
echo "Application finished."