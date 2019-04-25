import keras
import tensorflow as tf
from keras import backend as K
from keras import layers as L
from keras.layers import Layer
from keras.layers import Dense


class GCN(Layer):

    def __init__(self, output_dimensions, **kwargs):
        self.output_dim = output_dimensions
        super(GCN, self).__init__(**kwargs)
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feat_shape, adj_shape = input_shape
        self.kernel = self.add_weight(name='gcn_kernel', 
                                      shape=(int(feat_shape[2]),self.output_dim[1]),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(shape=(feat_shape[1],self.output_dim[1]),
                                    initializer='ones',
                                    name='gcn_bias',
                                    trainable=True)
        self.reshape_kernel = self.add_weight(name='gcn_reshape_kernel', 
                                      shape=(int(feat_shape[2]),self.output_dim[1]),
                                      initializer='uniform',
                                      trainable=True)
        super(GCN, self).build(feat_shape)
      
    def call(self, x):
        assert isinstance(x, list)
        X, A = x
        _X = tf.einsum('ijk,kl->ijl',X,self.kernel) + self.bias
        _X = tf.matmul(A,_X)
        _X = K.relu(self.skip(_X,X))
        
        print("layer output: ", _X)
        return list([_X,A])
    
    def skip(self, _X, X):
        if _X.shape[2] != X.shape[2]:
            _X = _X + K.dot(X,self.reshape_kernel)
        else:
            _X = _X+X
        return(_X)
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        feat_shape, adj_shape = input_shape
        return (self.output_dim)
