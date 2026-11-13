@echo off
rem Opens a maximised PowerShell window in the repository and starts the live demo (press Enter between steps).
cd /d "%~dp0..\.."
start "AFML live demo" /max powershell -NoExit -Command "cls; python docs\video\live_demo.py"
