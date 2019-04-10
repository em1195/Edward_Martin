import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense


class GGCN(Layer):

    def __init__(self, output_dimensions, **kwargs):
        self.output_dim = output_dimensions
        super(GGCN, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(int(input_shape[0][2]),self.output_dim[1]),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.output_dim[1]),
                                    initializer='ones',
                                    name='bias',
                                    trainable=True)
        self.update_bias = self.add_weight(shape=(self.output_dim[1]),
                                    initializer='ones',
                                    name='update_bias',
                                    trainable=True)
        
        super(GGCN, self).build(input_shape)
      
    def call(self, x):
        assert isinstance(x, list)
        X, A = x        
        _X = tf.einsum('ijk,kl->ijl',X,self.kernel) + self.bias
        _X = K.batch_dot(A,_X)
        _X = K.relu(_X)
        _X = self.gated_skip(_X,X)
        print("layer output: ", _X)
        return list([_X,A])
    
    def gated_skip(self, _X, X):
        if _X.shape != X.shape:
            X = Dense(self.output_dim[1])(X)
        _u = Dense(self.output_dim[1])(_X)
        u = Dense(self.output_dim[1])(X)
        
        z = K.sigmoid(_u + u + self.update_bias)
        
        _X = keras.layers.multiply([_X,z]) + keras.layers.multiply([X,1-z])
        return(_X)
        
    def compute_output_shape(self, input_shape):
        return (self.input_shape, self.output_dim[1])