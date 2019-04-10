import tensorflow as tf
from tensorflow.keras.layers import Layer

class Readout_NW(Layer):

    def __init__(self, output_dimensions, **kwargs):
       
        self.output_dim = output_dimensions
        super(Readout_NW, self).__init__(**kwargs)
        
     
    def build(self, input_shape):
        
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(int(input_shape[0][2]),self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
               
        self.bias = self.add_weight(shape=(512),
                                    initializer='ones',
                                    name='bias',
                                    trainable=True)
        super(Readout_NW, self).build(input_shape)

    def call(self, x):
        
        assert isinstance(x, list) 
        X, A = x        
        _X = tf.einsum('ijk,kl->ijl',X,self.kernel) + self.bias
        _X = tf.reduce_sum(_X, 1)
        print("readout output: ", _X)
        return _X

    def compute_output_shape(self, input_shape):
        return (self.input_shape, self.output_dim[1])