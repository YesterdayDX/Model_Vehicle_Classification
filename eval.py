from cmath import log
import os,random
import numpy as np
import tensorflow as tf
import argparse

from model import deepSense
from utils import load_dataset

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", help="length of data", \
        type=int, default=100)
    parser.add_argument("-m", "--model", help="weights of the trained neural network model", \
        type=str, default="./log/weight_default_1sec.h5")
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

    # Load the best weights once training is finished
    model = tf.keras.models.load_model(logfile)
    model.summary()

    print("=================================")
    print("Load Model from", logfile)
    print("=================================")
    # Show simple version of performance
    score = model.evaluate(X_test, Y_test, verbose=1, batch_size=64)
    print(score)


