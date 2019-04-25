import keras
import tensorflow as tf
from keras import backend as K
from keras import layers as L
from keras.layers import Layer
from keras.layers import Dense

class PREDICT(Layer):

    def __init__(self,output_dim, **kwargs):
        self.output_dim = output_dim
        super(PREDICT, self).__init__(**kwargs)
        
     
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(PREDICT, self).build((input_shape,self.output_dim))

    def call(self, x):
        X = x
        _X = K.dot(X, self.kernel)
        _X = K.relu(_X)
        _X = K.sum(_X, axis=1, keepdims=True)
        _X = K.sigmoid(_X)
        
        print("predict output: ", _X)
        return _X

    def compute_output_shape(self, input_shape):
        return (self.output_dim)
