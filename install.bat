@echo off
echo Setting up Fatigue Detection project...

echo Checking Python version...
python --version

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup completed successfully!
echo You can now run the application with: python -m src.fatigue_detection_app
echo.

pause