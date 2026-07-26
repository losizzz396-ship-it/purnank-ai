@echo off
title Purnank Launcher
cd /d "%~dp0"

echo Starting Purnank server...
start "Purnank Server (keep this open)" cmd /k python main.py

echo Waiting for it to boot...
timeout /t 6 /nobreak >nul

echo Opening Purnank in your browser...
start "" http://localhost:8000/

echo.
echo Purnank is running in the other window titled "Purnank Server".
echo Close that window (or press Ctrl+C in it) to stop the app.
echo This window can be closed.
pause
