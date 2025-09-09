@echo off
echo ===================================================
echo    Analisador de Imagens - Build Script (Windows)
echo ===================================================

echo.
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.x from https://python.org
    pause
    exit /b 1
)
python --version

echo.
echo [2/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [3/4] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip

echo Uninstalling existing PySimpleGUI...
pip uninstall PySimpleGUI -y
pip cache purge

echo Installing PySimpleGUI from private server...
pip install --upgrade --extra-index-url https://PySimpleGUI.net/install PySimpleGUI

echo Installing other dependencies...
pip install opencv-python numpy scikit-learn Pillow

echo.
echo [4/4] Starting the application...
cd src
python main.py

echo.
echo Application finished.
pause