import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense


class GCN(Layer):

    def __init__(self, output_dimensions, **kwargs):
        self.output_dim = output_dimensions
        super(GCN, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(int(input_shape[0][2]),self.output_dim[1]),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.output_dim[1]),
                                    initializer='ones',
                                    name='bias',
                                    trainable=True)
        super(GCN, self).build(input_shape)
      
    def call(self, x):
        assert isinstance(x, list)
        X, A = x        
        _X = tf.einsum('ijk,kl->ijl',X,self.kernel) + self.bias
        _X = K.batch_dot(A,_X)
        _X = self.skip(_X,X)
        print("layer output: ", _X)
        return list([_X,A])
    
    def skip(self, _X, X):
        if _X.shape != X.shape:
            _X = K.relu(_X + Dense(self.output_dim[1])(X))
        else:
            _X = K.relu(_X+X)
        return(_X)
        
    def compute_output_shape(self, input_shape):
        return (self.input_shape, self.output_dim[1])
