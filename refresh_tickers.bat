@echo off
cd /d "C:\Users\gaube\Cursor_Projects"
call venv\Scripts\activate.bat
python myutils.py refresh
echo.
echo Ticker refresh completed at %date% %time%
pause