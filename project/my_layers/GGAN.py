import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense


class GGAN(Layer):

    def __init__(self, output_dimensions,num_heads, **kwargs):
        self.output_dim = output_dimensions
        self.num_heads = num_heads
        super(GGAN, self).__init__(**kwargs)
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
        self.attn_list = []
        for i in range(self.num_heads):
            self.attn = self.add_weight(name='attention' + str(i),
                                        shape =(self.output_dim[1],self.output_dim[1]),
                                        initializer='uniform',
                                        trainable=True)
            self.attn_list.append(self.attn)  
        
        super(GGAN, self).build(input_shape)
      
    def call(self, x):
        assert isinstance(x, list)
        X, A = x        
        _X = tf.einsum('ijk,kl->ijl',X,self.kernel) + self.bias
        _X = K.batch_dot(A,_X)
        _X = self.calc_attn(_X, A)  
        _X = self.gated_skip(_X,X)
        print("layer output: ", _X)
        return list([_X,A])
    
    def calc_attn(self, _X, A):
        heads = []
        for i in range(self.num_heads):
            _X1 = K.batch_dot(_X,tf.einsum('ij,ajk->aik', self.attn_list[i], K.permute_dimensions(_X, (0,2,1)))) 
            _A = keras.layers.multiply([A,_X1])
            _A = K.tanh(_A)
            head = K.batch_dot(_A, _X)
            heads.append(head)
        _X = tf.reduce_mean(heads, 0)
        _X = K.relu(_X)
        return(_X)
    
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