import os,random
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Conv1D,Conv2D,Dropout,concatenate,Reshape,Lambda,LSTM, Flatten, MaxPool2D
from scipy import signal
from utils import load_dataset

def convert_to_lite_model(logfile, logfile_lite):
    model = tf.keras.models.load_model(logfile)
    model.summary()

    batch_size = 1
    input_shape = model.inputs[0].shape.as_list()
    input_shape[0] = batch_size
    func1 = tf.function(model).get_concrete_function([tf.TensorSpec(input_shape, model.inputs[0].dtype, name="acoustic"), tf.TensorSpec(input_shape, model.inputs[0].dtype, name="seismic")])
    model_converter = tf.lite.TFLiteConverter.from_concrete_functions([func1])

    model_lite = model_converter.convert()
    # # # Store the Lite Model
    with open(logfile_lite, 'wb') as f:
        print("Write lite model to "+logfile_lite)
        f.write(model_lite)
    return

def test_lite_model(L, logfile_lite):
    print("========= Testing Lite Model ===========")
    X_train_a, X_train_s, X_val_a, X_val_s, X_test_a, X_test_s, Y_train, Y_val, Y_test = load_dataset(L)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=logfile_lite)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    N = 3

    for i in range(N):
        x_a = X_test_a[i]
        x_s = X_test_s[i]
        x_a = x_a[np.newaxis,...,np.newaxis]
        x_s = x_s[np.newaxis,...,np.newaxis]

        x_a = np.float32(x_a)
        x_s = np.float32(x_s)
        # print(x_a.shape)

        interpreter.set_tensor(input_details[0]['index'], x_a)
        interpreter.set_tensor(input_details[1]['index'], x_s)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        print(Y_test[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", help="length of data", \
        type=int, default=100)
    parser.add_argument("-m", "--model", help="weights of the trained neural network model", \
        type=str, default="./log/weight_default_1sec.h5")
    args = parser.parse_args()

    logfile = args.model
    logfile_lite = "./liteModel/deepsense.tflite"

    convert_to_lite_model(logfile, logfile_lite)
    test_lite_model(args.length, logfile_lite)


