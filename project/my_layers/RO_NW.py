import keras
import tensorflow as tf
from keras import backend as K
from keras import layers as L
from keras.layers import Layer
from keras.layers import Dense

class Readout_NW(Layer):

    def __init__(self, output_dimensions, **kwargs):
       
        self.output_dim = int(output_dimensions)
        super(Readout_NW, self).__init__(**kwargs)
        
     
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1],self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(shape=(input_shape[0], self.output_dim),
                                    initializer='ones',
                                    name='bias',
                                    trainable=True)
        super(Readout_NW, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list) 
        X, A = x
        _X = tf.einsum('ijk,kl->ijl',X,self.kernel)
        _X = _X + self.bias
        #_X = K.dot(X,self.kernel) + self.bias
        _X = K.relu(_X)
        _X = tf.reduce_sum(_X, 1)
        _X = K.sigmoid(_X)
        print("readout output: ", _X)
        return _X

    def compute_output_shape(self, input_shape):
        return (self.output_dim)
