@echo off
echo ========================================
echo Kyutai STT & TTS - Simple Setup
echo ========================================

REM Setup portable paths
set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1%
set HF_HOME=%PROJECT_DIR%\cache
set TEMP=%PROJECT_DIR%\temp
set TMP=%PROJECT_DIR%\temp

REM Create directories
mkdir cache 2>nul
mkdir temp 2>nul

echo Project: %PROJECT_DIR%
echo Cache: %HF_HOME%
echo Temp: %TEMP%

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Install packages
echo Installing packages...
venv\Scripts\pip install --upgrade pip

REM Install PyTorch with CUDA (force reinstall)
echo Installing PyTorch with CUDA...
venv\Scripts\pip uninstall -y torch torchvision torchaudio
venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

REM Test CUDA
echo Testing CUDA...
venv\Scripts\python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

REM Install other packages
echo Installing other packages...
venv\Scripts\pip install gradio>=4.0.0
venv\Scripts\pip install moshi>=0.2.11
venv\Scripts\pip install numpy sounddevice sphn julius huggingface_hub sentencepiece soundfile pydub psutil

REM Download voices for TTS
echo Downloading voice library...
venv\Scripts\python download_voices.py

REM Remove conflicting packages
echo Removing conflicting packages...
venv\Scripts\pip uninstall -y TTS coqui-tts 2>nul

REM Set environment variable to disable symlink warning
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo.
echo ========================================
echo Setup completed!
echo ========================================
echo.
echo Starting WebUI...
venv\Scripts\python webui.py

pause