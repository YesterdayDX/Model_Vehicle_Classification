import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(2009)

def load_dataset(train_rate=1.0, dataset='split_run'):
    df = pd.read_csv('./train_data/train_X_both.csv', header=None)
    X_train = df.values.reshape(df.shape[0],2,200).astype('float64')
    df = pd.read_csv('./train_data/val_X_both.csv', header=None)
    X_val = df.values.reshape(df.shape[0],2,200).astype('float64')
    df = pd.read_csv('./train_data/test_X_both.csv', header=None)
    X_test = df.values.reshape(df.shape[0],2,200).astype('float64')
    df = pd.read_csv('./train_data/train_Y.csv', header=None)
    Y_train = df.values
    df = pd.read_csv('./train_data/val_Y.csv', header=None)
    Y_val = df.values
    df = pd.read_csv('./train_data/test_Y.csv', header=None)
    Y_test = df.values

    X_train=np.swapaxes(X_train,1,2)
    X_val=np.swapaxes(X_val,1,2)
    X_test=np.swapaxes(X_test,1,2)

    # Normalization
    for i in range(X_train.shape[0]):
        for j in range(X_train[i].shape[1]):
            m = np.max(np.absolute(X_train[i,:,j]))
            X_train[i,:,j] = X_train[i,:,j]/m

    for i in range(X_val.shape[0]):
        for j in range(X_val[i].shape[1]):
            m = np.max(np.absolute(X_val[i,:,j]))
            X_val[i,:,j] = X_val[i,:,j]/m

    for i in range(X_test.shape[0]):
        for j in range(X_test[i].shape[1]):
            m = np.max(np.absolute(X_test[i,:,j]))
            X_test[i,:,j] = X_test[i,:,j]/m

    # Seperate the acoustic and seismic
    X_train_a = X_train[:,:,0]
    X_val_a = X_val[:,:,0]
    X_test_a = X_test[:,:,0]
    X_train_s = X_train[:,:,1]
    X_val_s = X_val[:,:,1]
    X_test_s = X_test[:,:,1]

    return X_train_a, X_train_s, X_val_a, X_val_s, X_test_a, X_test_s, Y_train, Y_val, Y_test