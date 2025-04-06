# Fatigue Detection System

An AI-powered fatigue detection system that monitors eye and facial features to detect signs of fatigue in real-time.

## Features

- Real-time fatigue detection using computer vision
- Multiple detection algorithms (Eye Aspect Ratio, Mouth Aspect Ratio, Head Pose Estimation)
- User-friendly Gradio interface
- Configurable detection thresholds

## Requirements

- Python 3.11.x
- Dependencies listed in `requirements.txt`

## Installation

### Option 1: Using the setup script (Recommended)

```bash
python setup.py
```

This script will:
1. Verify Python version compatibility
2. Install all required dependencies

### Option 2: Manual installation

```bash
pip install -r requirements.txt
```

### Option 3: Using Poetry

If you prefer using Poetry for dependency management:

```bash
poetry install
```

To install without GPU support:

```bash
poetry install --without gpu
```

## Running the Application

```bash
python -m src.fatigue_detection_app
```

## Testing

Run the test suite with:

```bash
pytest
```

## Project Structure

```
src/
├── core/                  # Business logic
│   ├── detection/         # Algorithm implementations
│   └── models/            # ML model management
├── interface/             # UI components
├── utils/                 # Helper functions
├── config.py              # Settings management
└── fatigue_detection_app.py  # Main application entry point
tests/                     # Unit tests
```

## License

[MIT](LICENSE)