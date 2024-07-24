@echo off
cls
call D:\anaconda3\Scripts\activate.bat pytorch

setlocal enabledelayedexpansion
set /a count=10

echo ==================================All Batch training started.=========================================
for /l %%i in (1, 1, %count%) do (
    echo ==============================Performing the first %%i Batch training...==============================
    python dqn.py
    echo ================================Clause %%i Batch training completed.==================================
)
echo ==================================All Batch training completed.=======================================
pause
