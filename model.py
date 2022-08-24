import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Conv1D,Conv2D,Dropout,concatenate,Reshape,Lambda,LSTM


def deepSense(weights=None,
             input_shape1=[100,1],
             input_shape2=[100,1],
             classes=3):

    dr=0.3
    r=1e-4

    input1=Input(input_shape1,name='Acoustic')
    input2=Input(input_shape2,name='Seismic')

    x1 = Conv1D(64, 4,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=r))(input1)
    x2 = Conv1D(64, 4,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=r))(input2)
    x=concatenate([x1,x2],axis=2,name='Concatenate1')
    x = Conv1D(128, 4,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = LSTM(units=128,return_sequences=True,name="LSTM1",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = LSTM(units=128,return_sequences=True,name="LSTM2",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = Conv1D(128, 8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    x=Dense(128,activation="relu",name="FC1")(x)
    x=Dropout(dr)(x)
    x=Dense(classes,activation="softmax",name="Softmax")(x)

    model=Model(inputs=[input1,input2],outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)
    
    return model