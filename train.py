from cmath import log
import os,random
import numpy as np
import tensorflow as tf
import argparse
import time

from model import deepSense
from utils import load_dataset

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="weights of the trained neural network model", \
        type=str, default="./log/weight_default.h5")
    args = parser.parse_args()

    logfile = args.model

    # Load Dataset
    X_train_a, X_train_s, X_val_a, X_val_s, X_test_a, X_test_s, Y_train, Y_val, Y_test = load_dataset()

    print(X_train_a.shape, X_train_s.shape, Y_train.shape)
    print(X_val_a.shape, X_val_s.shape, Y_val.shape)
    print(X_test_a.shape, X_test_s.shape, Y_test.shape)

    # Train the model with both acoustic and seismic data
    X_train = [X_train_a, X_train_s]
    X_val = [X_val_a, X_val_s]
    X_test = [X_test_a, X_test_s]

    # DeepSense Model
    model = deepSense()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    print(logfile)

    start = time.process_time()
    # Train the model
    history = model.fit(X_train,
        Y_train,
        batch_size=64,
        epochs=100,
        verbose=2,
        validation_data=(X_val,Y_val),
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(logfile, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,verbose=1,patince=5,min_lr=0.0000001),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto'),           
                    ]
                        )
    end = time.process_time()
    print("\n=========================")
    print("Training takes",end-start, "seconds.")
    print("=========================\n")
                    

    # Show simple version of performance
    model = tf.keras.models.load_model(logfile)
    score = model.evaluate(X_test, Y_test, verbose=1, batch_size=64)
    print(score)


