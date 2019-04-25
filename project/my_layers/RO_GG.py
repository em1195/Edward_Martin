import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Readout_GG(Layer):

    def __init__(self, output_dimensions, **kwargs):
       
        self.output_dim = int(output_dimensions)
        super(Readout_GG, self).__init__(**kwargs)
        
     
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feat1_shape, feat2_shape = input_shape
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(int(feat_shape[2]),self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
               
        self.bias = self.add_weight(shape=(feat_shape[1], self.output_dim),
                                    initializer='ones',
                                    name='bias',
                                    trainable=True)
        super(Readout_GG, self).build(feat_shape)

    def call(self, x):
        assert isinstance(x, list) 
        X, A = x
        _X = tf.einsum('ijk,kl->ijl',X,self.kernel)+self.bias
        #_X = K.dot(X,self.kernel) + self.bias
        _X = K.relu(_X)
        _X = tf.reduce_sum(_X, 1)
        _X = K.sigmoid(_X)
        print("readout output: ", _X)
        return _X

    def compute_output_shape(self, input_shape):
        return (self.input_shape, self.output_dim)
