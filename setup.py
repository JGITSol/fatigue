import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements (>=3.11,<3.12)"""
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 11 and sys.version_info.minor < 12):
        print(f"Error: Python 3.11.x is required. Current version: {sys.version}")
        sys.exit(1)
    print(f"Python version check passed: {sys.version}")

def install_dependencies():
    """Install dependencies from requirements.txt"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("Error: requirements.txt not found")
        sys.exit(1)
        
    print("Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
            check=True
        )
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def main():
    print("Setting up Fatigue Detection project...")
    check_python_version()
    install_dependencies()
    print("\nSetup completed successfully!")
    print("You can now run the application with: python -m src.fatigue_detection_app")

if __name__ == "__main__":
    main()