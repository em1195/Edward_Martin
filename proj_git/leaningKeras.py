import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import random
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

 
class MyDense(Layer):

    def __init__(self, **kwargs):
        print("initialising.....")
        self.output_dim = (50,32)
        super(MyDense, self).__init__(**kwargs)
        print("INITIALISED")
        
     
    def build(self, input_shape):
        print("building.....")
        print("creating weights.....")
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(58,32),
                                      initializer='uniform',
                                      trainable=True)
        print("WEIGHTS BUILT")
        print("building biases.....")
        
        self.bias = self.add_weight(shape=(32),
                                    initializer='zeros',
                                    name='bias',
                                    trainable=True)
        print("BIASES BUILT")
        super(MyDense, self).build(input_shape)
        print("BUILT")

    def call(self, x):
        print("calling.....")
        print(x.shape)
        _x = tf.einsum('ijk,kl->ijl',x,self.kernel) + self.bias
        print(_x.shape)
        print("FINISHED CALLING")
        
        
        #y = K.dot(x, self.kernel)
        return _x

    def compute_output_shape(self, input_shape):
        return (self.input_shape, self.output_dim[1])
    

adj = np.load("A0.npy")
feat = np.load("F0.npy")

X = K.variable(feat)
A = K.variable(adj[0])
input_shapes = feat.shape
print("creating inputs")
inputs = Input(shape=(50,58),batch_size=10000)
print("INPUTS CREATED")
print(inputs)
print("entering custom layer")
x = MyDense(input_shape=input_shapes)(inputs)
print("EXITING CUSTOM LAYER")
print(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(32, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
