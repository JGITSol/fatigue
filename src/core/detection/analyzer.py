from __future__ import annotations
import cv2
import numpy as np
from loguru import logger
from ...config import AppConfig, AlgorithmType
# Plain ASCII art without formatting markers
APP_ASCII = """
███████╗ █████╗ ████████╗██╗ ██████╗ ██╗   ██╗███████╗
██╔════╝██╔══██╗╚══██╔══╝██║██╔════╝ ██║   ██║██╔════╝
█████╗  ███████║   ██║   ██║██║  ███╗██║   ██║█████╗  
██╔══╝  ██╔══██║   ██║   ██║██║   ██║██║   ██║██╔══╝  
██║     ██║  ██║   ██║   ██║╚██████╔╝╚██████╔╝███████╗
╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝
"""
class FatigueAnalyzer:
    """Core analysis engine with strategy pattern"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._load_models()
        self._step_counter = 0
        self._current_pipeline = []
        
    def _load_models(self):
        """Load ML models with error handling"""
        self.models = {
            AlgorithmType.EAR: self._init_tflite_model(),
            AlgorithmType.MAR: self._init_mediapipe(),
            AlgorithmType.HEAD_POSE: self._init_head_pose()
        }
        
    def _init_tflite_model(self):
        """Initialize TensorFlow Lite model for eye aspect ratio detection"""
        try:
            # Import TensorFlow and TFLite correctly
            import tensorflow as tf
            logger.info("Initializing TFLite model...")
            interpreter = tf.lite.Interpreter(
                model_path=str(self.config.model_path)
            )
            # Allocate memory for the model
            interpreter.allocate_tensors()
            logger.info("TFLite model initialized successfully")
            return interpreter
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            # Return None instead of raising to allow the app to continue
            logger.warning("Using fallback for EAR detection")
            return None
            
    def _init_mediapipe(self):
        """Initialize MediaPipe for mouth aspect ratio detection"""
        try:
            # Import mediapipe only when needed to avoid unnecessary dependencies
            import mediapipe as mp
            logger.info("Initializing MediaPipe for MAR detection...")
            # Return a simple face mesh solution
            mp_face_mesh = mp.solutions.face_mesh
            return mp_face_mesh.FaceMesh(static_image_mode=False, 
                                         max_num_faces=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        except ImportError:
            logger.warning("MediaPipe not available, using fallback for MAR")
            return None
        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {str(e)}")
            return None
    
    def _init_head_pose(self):
        """Initialize head pose estimation model"""
        try:
            logger.info("Initializing head pose estimation...")
            # For simplicity, we'll use a basic OpenCV-based approach
            # In a real implementation, this would be a more sophisticated model
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            return face_cascade
        except Exception as e:
            logger.error(f"Head pose model initialization failed: {str(e)}")
            return None

    def analyze_frame(self, frame: np.ndarray) -> dict:
        """Main processing pipeline"""
        try:
            preprocessed = self._preprocess(frame)
            features = self._extract_features(preprocessed)
            return self._detect_fatigue(features)
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"error": str(e), "status": "failed"}

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Frame preprocessing pipeline"""
        pipeline = [
            ('Resize', lambda x: cv2.resize(x, self.config.resolution)),
            ('Normalize', lambda x: x.astype(np.float32) / 255.0),
            ('Convert Color', lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        ]
        
        for name, op in pipeline:
            frame = op(frame)
        return frame
        
    def _extract_features(self, frame: np.ndarray) -> dict:
        """Extract relevant features based on the selected algorithm"""
        features = {}
        
        if self.config.algorithm == AlgorithmType.EAR:
            features["ear"] = self._calculate_ear(frame)
        elif self.config.algorithm == AlgorithmType.MAR:
            features["mar"] = self._calculate_mar(frame)
        elif self.config.algorithm == AlgorithmType.HEAD_POSE:
            features["head_pose"] = self._calculate_head_pose(frame)
        
        return features
    
    def _calculate_ear(self, frame: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio using TFLite model"""
        # Check if TFLite model is available
        if self.models[AlgorithmType.EAR] is None:
            # Fallback to a placeholder implementation if model is not available
            logger.warning("Using fallback EAR calculation")
            return 0.3  # Normal EAR value
            
        try:
            # Prepare input data for the model
            interpreter = self.models[AlgorithmType.EAR]
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Resize frame to match model input shape
            input_shape = input_details[0]['shape'][1:3]
            input_data = cv2.resize(frame, input_shape)
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
            
            # Set input tensor and invoke model
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])
            ear_value = float(output_data[0][0])
            
            return ear_value
        except Exception as e:
            logger.error(f"Error calculating EAR: {str(e)}")
            return 0.3  # Fallback to normal EAR value
    
    def _calculate_mar(self, frame: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio using MediaPipe"""
        # Placeholder implementation
        return 0.2  # Normal MAR value
    
    def _calculate_head_pose(self, frame: np.ndarray) -> dict:
        """Calculate head pose angles"""
        # Placeholder implementation
        return {"pitch": 0, "yaw": 0, "roll": 0}
    
    def _detect_fatigue(self, features: dict) -> dict:
        """Detect fatigue based on extracted features"""
        result = {"fatigue_detected": False, "confidence": 0.0, "features": features}
        
        if "ear" in features and features["ear"] < self.config.threshold_ear:
            result["fatigue_detected"] = True
            result["confidence"] = 1.0 - (features["ear"] / self.config.threshold_ear)
        
        if "mar" in features and features["mar"] > self.config.threshold_mar:
            result["fatigue_detected"] = True
            result["confidence"] = features["mar"] / self.config.threshold_mar
        
        return result