# test_imports.py
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler

def main():
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)
    print("All TensorFlow and Keras imports are working correctly.")

if __name__ == "__main__":
    main()
