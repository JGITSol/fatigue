from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, AnyHttpUrl, field_validator, AnyUrl, HttpUrl, SecretStr, SecretBytes, EmailStr
from pydantic_settings import BaseSettings

# Plain ASCII art without formatting markers
APP_ASCII = """
███████╗ █████╗ ████████╗██╗ ██████╗ ██╗   ██╗███████╗
██╔════╝██╔══██╗╚══██╔══╝██║██╔════╝ ██║   ██║██╔════╝
█████╗  ███████║   ██║   ██║██║  ███╗██║   ██║█████╗  
██╔══╝  ██╔══██║   ██║   ██║██║   ██║██║   ██║██╔══╝  
██║     ██║  ██║   ██║   ██║╚██████╔╝╚██████╔╝███████╗
╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝
"""
class AlgorithmType(str, Enum):
    EAR = "eye_aspect_ratio"
    MAR = "mouth_aspect_ratio"
    HEAD_POSE = "head_pose_estimation"

class AppConfig(BaseSettings):
    model_path: Path = Path("models/optimized.tflite")
    threshold_ear: float = 0.25
    threshold_mar: float = 0.4
    frame_skip: int = 2
    resolution: tuple = (640, 480)
    algorithm: AlgorithmType = AlgorithmType.EAR
    debug_mode: bool = False

    class Config:
        env_prefix = "FDA_"
        case_sensitive = False
