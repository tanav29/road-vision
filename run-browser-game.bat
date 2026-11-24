@echo off
cd /d "%~dp0testing"
"%~dp0.venv\Scripts\python.exe" webcam.py --screen
if errorlevel 130 exit /b %errorlevel%
pause
