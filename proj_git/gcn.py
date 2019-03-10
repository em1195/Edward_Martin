import numpy as np
import tensorflow as tf
import os

X_new = np.load('X0.npy')
A_new = np.load('A0.npy')

class gcn():
    
    def __init__(self):
        self.X = tf.placeholder(tf.float64, shape=(self.batch_size, 50,50))
        self.A = tf.placeholder(tf.float64, shape=(self.batch_size, 50, 58))
        self.Y = tf.placeholder(tf.float64, shape=(self.batch_size))
        print('initialised')
        
        
    def create_network(self):
        self.X = tf.cast(self.X, tf.float64)
        self.A = tf.cast(self.A, tf.float64)
        self.Y = tf.cast(self.Y, tf.float64)
        self._X = None
        self._A = None
        self.Z = None
        
    def init_input(self, X, A, layers):
        num_atoms = int(X.get_shape()[1])
        dim_in = int(X.get_shape()[2])
        dim_hidden = []
        dim_hideen.append(dim_in)
        for i in range(layers):
            dim_hidden.append(32)
            
        _X = X
        
        for i in range((dim_hidden)-1):
            weights = tf.get_variable("w", shape = [dim_hidden[i],dim_hidden[i+1]], dtype=tf.float64)
            biases = tf.get_variable("b", shape = [dim_hidden[i],dim_hidden[i+1]], dtype=tf.float64, initializer=zeros_initializer)
        
        
        
        
    
        
        