import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import random
import os

def create_structure(X, A, num_layers):
    dim_input = int(X.get_shape()[2])
    num_atoms = int(X.get_shape()[1])
    
    dim_list = [dim_input]
    for i in range(num_layers-1):
        dim_list.append(32)
    
    _X = X
    
    for i in range(num_layers):
        w = tf.get_variable("w"+str(i),[dim_list[i],32],dtype=tf.float32, initializer = tf.initializers.glorot_uniform)
        b = tf.get_variable("b"+str(i), [32],dtype=tf.float32, initializer = tf.initializers.glorot_uniform)
        tf.add_to_collection("weights", w)
        tf.add_to_collection("biases", b)
        print("MADE WEIGHT LAYER ", str(i))
    
        _X = conv(A,_X,w,b)
    
    return(_X)
    
def conv(A,X,w,b):
    
    _X = tf.einsum('ijk,kl->ijl',X,w) + b
    _X = tf.matmul(A, _X)
    
    return _X

tf.keras.backend.clear_session()

adj = np.load("A0.npy")
feat = np.load("F0.npy")

X = K.variable(feat)
A = K.variable(adj)

new_X = create_structure(X,A,10)


