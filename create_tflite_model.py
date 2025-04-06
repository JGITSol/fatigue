import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# Create a simple model for eye aspect ratio detection
# This is a placeholder model with appropriate structure
def create_model():
    # Simple model that takes an image input and outputs a single value (EAR)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(24, 24, 1)),  # Small grayscale image input
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output EAR value between 0-1
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and save the model
def main():
    print("Creating TensorFlow Lite model...")
    
    # Create the model
    model = create_model()
    
    # Create a sample input for testing
    sample_input = np.random.rand(1, 24, 24, 1).astype(np.float32)
    
    # Test prediction
    prediction = model.predict(sample_input)
    print(f"Test prediction: {prediction[0][0]}")
    
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TFLite model
    model_dir = Path("models")
    model_path = model_dir / "optimized.tflite"
    
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {model_path}")
    print("Model size: {:.2f} KB".format(os.path.getsize(model_path) / 1024))

if __name__ == "__main__":
    main()