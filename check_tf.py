try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("TensorFlow successfully imported!")
except Exception as e:
    print(f"Error importing TensorFlow: {e}")