import pyaudio
import wave
import numpy as np
import time,datetime,os,csv
import matplotlib.pyplot as plt
import pandas as pd
import ast
import math
import random
import os
import geopy.distance

def get_data(path, step, shift=0):
    df = pd.read_csv(DIR+path+'/cut/ehz.csv', sep = ' ', header=None, dtype=np.float64)
    x_ehz = df.values
    df = pd.read_csv(DIR+path+'/cut/aud.csv', sep = ' ', header=None, dtype=np.float64)
    x_aud = df.values
    df = pd.read_csv(DIR+path+'/cut/aud16000.csv', sep = ' ', header=None, dtype=np.float64)
    x_aud16000 = df.values.flatten()

    x_t = x_aud[:,1]
    x_ehz = x_ehz[:,0]
    x_aud = x_aud[:,0]

    X_both, X_sei, X_aud_100, X_aud_16000, Y, Y_16000, Dist = [], [], [], [], [], [], []

    if path[0] == 'P':
        y = [1,0,0]
    elif path[0] == 'S':
        y = [0,1,0]
    else:
        y = [0,0,1]

    ind = 0
    while ind + step + shift < x_ehz.shape[0]:
        x = np.array([x_aud[ind+shift:ind+step+shift],x_ehz[ind:ind+step]])
        x = x.reshape((step*2, ))
        X_aud_100.append(x_aud[ind+shift:ind+step+shift])
        X_sei.append(x_ehz[ind:ind+step])
        X_both.append(x)
        Y.append(y)
        ind += int(step/2)

    ind = 0
    while ind + step*160 < x_aud16000.shape[0]:
        X_aud_16000.append(x_aud16000[ind:ind+step*160])
        Y_16000.append(y)
        ind += step*80

    X_aud_100 = np.array(X_aud_100)
    X_sei = np.array(X_sei)
    X_both = np.array(X_both)
    X_aud_16000 = np.array(X_aud_16000)
    Y = np.array(Y)

    return X_sei, X_aud_100, X_aud_16000, X_both, Y


PATH = ["Polaris0150pm", "Polaris0215pm-AllChannels", "Polaris0235pm-NoLineOfSight", \
 "Silverado0255pm", "Silverado0315pm-AllChannels", "Warhog-NoLineOfSight", \
"Warhog1135am", "Warhog1149am", "Warhog1209am-AllChannels"]
DIR = "/data/dongxin3/data/GQ-2022-01-06/"

step = 200

X_train_both = np.array([])
X_val_both = np.array([])
X_test_both = np.array([])
Y_train = np.array([])
Y_val = np.array([])
Y_test = np.array([])

# Training and Validation
for i in [0, 1, 3, 5, 6, 7]:
    path = PATH[i]
    X_sei, X_aud_100, X_aud_16000, X_both, Y = get_data(path, step)
    if X_train_both.shape[0] == 0:
        X_train_both = X_both
        Y_train = Y
    else:
        X_train_both = np.concatenate((X_train_both, X_both), axis=0)
        Y_train = np.concatenate((Y_train, Y), axis=0)
    print(path, X_both.shape)

# Testing
for i in [2, 4, 8]:
    path = PATH[i]
    X_sei, X_aud_100, X_aud_16000, X_both, Y = get_data(path, step)

    if X_test_both.shape[0] == 0:
        X_test_both = X_both
        Y_test = Y
    else:
        X_test_both = np.concatenate((X_test_both, X_both), axis=0)
        Y_test = np.concatenate((Y_test, Y), axis=0)
    print(path, X_sei.shape)



print(X_train_both.shape, X_test_both.shape)

index = [i for i in range(X_train_both.shape[0])]
train_index = random.sample(index, int(len(index)*0.7))
val_index = list(set(index)-set(train_index))

print(len(index), len(train_index), len(val_index))
# X_val_sei = X_train_sei[val_index]
# X_val_aud = X_train_aud[val_index]
X_val_both = X_train_both[val_index]
Y_val = Y_train[val_index]
# D_val = D_train[val_index]

# X_train_sei = X_train_sei[train_index]
# X_train_aud = X_train_aud[train_index]
X_train_both = X_train_both[train_index]
Y_train = Y_train[train_index]
# D_train = D_train[train_index]

print(X_train_both.shape, X_val_both.shape, X_test_both.shape)
print(Y_train.shape, Y_val.shape, Y_test.shape)
# print(D_train.shape, D_val.shape, D_test.shape)
print(np.sum(Y_train, axis=0))
print(np.sum(Y_val, axis=0))
print(np.sum(Y_test, axis=0))
# print(min(D_train), min(D_val), min(D_test))

# print(X_train_both.shape, X_val_both.shape, X_test_both.shape, X_train_aud.shape)

folder_name = './train_data_'+str(int(step/100))+'sec/'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# df = pd.DataFrame(X_train_aud)
# df.to_csv(folder_name+'train_X_aud.csv', header=False, index=False)
df = pd.DataFrame(X_train_both)
df.to_csv(folder_name+'train_X_both.csv', header=False, index=False)
df = pd.DataFrame(Y_train)
df.to_csv(folder_name+'train_Y.csv', header=False, index=False)

# df = pd.DataFrame(X_test_aud)
# df.to_csv(folder_name+'test_X_aud.csv', header=False, index=False)
df = pd.DataFrame(X_test_both)
df.to_csv(folder_name+'test_X_both.csv', header=False, index=False)
df = pd.DataFrame(Y_test)
df.to_csv(folder_name+'test_Y.csv', header=False, index=False)
# df = pd.DataFrame(D_test)
# df.to_csv('./train_data/test_D.csv', header=False, index=False)

# df = pd.DataFrame(X_val_aud)
# df.to_csv(folder_name+'val_X_aud.csv', header=False, index=False)
df = pd.DataFrame(X_val_both)
df.to_csv(folder_name+'val_X_both.csv', header=False, index=False)
df = pd.DataFrame(Y_val)
df.to_csv(folder_name+'val_Y.csv', header=False, index=False)