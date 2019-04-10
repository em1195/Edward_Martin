import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from my_layers.GCN import GCN
from my_layers.GAN import GAN
from my_layers.RO_NW import Readout_NW
from my_layers.GGCN import GGCN
from my_layers.GGAN import GGAN

def use_GCN(num_layers, inputs, output_dim):
    layers_list = []
    x = GCN(output_dim,input_shape=input_shapes)(inputs)
    for i in range(int(num_layers)-1):
        x = GCN(output_dim,input_shape=input_shapes)(x)
        layers_list.append(x)
    z = Readout_NW(512,input_shape=input_shapes)(x)
    layers_list.append(z)
    y = Dense(1,activation="sigmoid")(z)
    layers_list.append(y)
    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

def use_GAN(num_layers, inputs, output_dim, heads):
    layers_list = []
    x = GAN(output_dim,heads,input_shape=input_shapes)(inputs)
    for i in range(int(num_layers)-1):
        x = GAN(output_dim,heads,input_shape=input_shapes)(x)
        layers_list.append(x)
    z = Readout_NW(512,input_shape=input_shapes)(x)
    layers_list.append(z)
    y = Dense(1,activation="sigmoid")(z)
    layers_list.append(y)
    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

def use_GGCN(num_layers, inputs, output_dim):
    layers_list = []
    x = GGCN(output_dim,input_shape=input_shapes)(inputs)
    for i in range(int(num_layers)-1):
        x = GGCN(output_dim,input_shape=input_shapes)(x)
        layers_list.append(x)
    z = Readout_NW(512,input_shape=input_shapes)(x)
    layers_list.append(z)
    y = Dense(1,activation="sigmoid")(z)
    layers_list.append(y)
    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

def use_GGAN(num_layers, inputs, output_dim,heads):
    layers_list = []
    x = GGAN(output_dim,heads,input_shape=input_shapes)(inputs)
    for i in range(int(num_layers)-1):
        x = GGAN(output_dim,heads,input_shape=input_shapes)(x)
        layers_list.append(x)
    z = Readout_NW(512,input_shape=input_shapes)(x)
    layers_list.append(z)
    y = Dense(1,activation="sigmoid")(z)
    layers_list.append(y)
    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    
tf.keras.backend.clear_session()
adj = np.load("A0.npy")
feat = np.load("F0.npy")
logP = np.load("logP.npy")
X = K.variable(feat)
A = K.variable(adj)
input_shapes = feat.shape
inputX = Input(shape=(int(X.shape[1]),int(X.shape[2])),batch_size=int(X.shape[0]))
inputA = Input(shape=(int(A.shape[1]),int(A.shape[2])),batch_size=int(A.shape[0]))
inputs = [inputX, inputA]
output_dim = [50,32]
   

print("Which network will you use?")
print("1: Graph Convolution Network")
print("2: Graph Convolution Network + attention")
print("3: Graph Convolution Network + gated skip connection")
print("4: Graph Convolution Network + attention + gated skip connection")
net_type = int(input("-- "))
num_layers = input("Number of layers: ")

if net_type == 1:
    use_GCN(num_layers, inputs, output_dim)
elif net_type == 2:
    num_heads = int(input("Enter number of heads: "))
    use_GAN(num_layers, inputs, output_dim, num_heads)
elif net_type == 3:
    use_GGCN(num_layers, inputs, output_dim)
elif net_type == 4:
    num_heads = int(input("Enter number of heads: "))
    use_GGAN(num_layers, inputs, output_dim, num_heads)
