@echo off
:: Chuyen bang ma sang UTF-8 de hien thi tieng Viet tren Terminal
chcp 65001 >nul
title VSL AI Translator - Khoi Dong Nhanh

echo ===================================================
echo      VSL AI TRANSLATOR - KHOI DONG HE THONG
echo ===================================================
echo.

:: Kiem tra xem Python da duoc cai dat chua
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python! Vui long cai dat Python va check vao o "Add to PATH".
    pause
    exit /b
)

:: Kiem tra va tao moi truong ao (virtual environment) tranh xung dot thu vien
if not exist "venv\" (
    echo [*] Dang tao moi truong ao (Virtual Environment)...
    python -m venv venv
)

:: Kich hoat moi truong ao va cai dat thu vien
echo [*] Dang kiem tra va cai thu vien (chi mat thoi gian o lan chay dau tien)...
call venv\Scripts\activate
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

:: Chay ung dung
echo.
echo [*] Moi thu da san sang! Dang khoi dong Web Server...
python app.py

pause