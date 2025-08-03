@echo off
echo ========================================
echo TED Talk Processor - Quick Start
echo ========================================
echo.

echo Step 1: Installing dependencies...
pip install -r requirements.txt

echo.
echo Step 2: Testing installation...
python test_installation.py

echo.
echo Step 3: Running example...
echo Processing a sample TED Talk...
python example.py

echo.
echo ========================================
echo Setup complete! 
echo ========================================
echo.
echo To process your own TED Talks:
echo   python ted_processor.py "YOUR_YOUTUBE_URL"
echo.
echo To process multiple videos from a file:
echo   python batch_process.py sample_urls.txt
echo.
pause 