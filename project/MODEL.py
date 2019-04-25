import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from my_layers.GCN import GCN
from my_layers.GAN import GAN
from my_layers.RO_NW import Readout_NW
from my_layers.RO_GG import Readout_GG
from my_layers.GGCN import GGCN
from my_layers.GGAN import GGAN
from my_layers.PREDICT import PREDICT
import os
from statistics import mean
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


def use_GCN(num_layers, inputs, output_dim):
    x = GCN(output_dim,input_shape=input_shapes)(inputs)
    print("output from first layer: ", x)
    z = Readout_NW(512,input_shape=input_shapes)(x)
    print("output from readout layer: ", z)
    #y = Dense(512,activation = 'relu', use_bias=False)(z)
    #y = Dense(512,activation='tanh', use_bias=False)(y)
    #y = Dense(1, use_bias = True)(y)
    y = PREDICT(1,input_shape=z.shape)(z)
    print("shape of final output: ", y.shape)
    
    model = Model(inputs=inputs, outputs=y)
    sgd = optimizers.SGD(lr=0.001, decay=0.5, momentum=0.9, nesterov=True)
    model.compile(optimizer = sgd, loss='mean_absolute_error',metrics=['mean_absolute_error'])
    logP = np.load(cwd + "/database/ZINC/logP.npy")
    plot_model(model, to_file='model.png')
    features = np.load(cwd + "/database/ZINC/features/" + str(0) + ".npy")
    adjs = np.load(cwd + "/database/ZINC/adj/" + str(0) + ".npy")
    num_files = 3
    epochs = 5
    print(model.weights)
    for i in range(num_files-1):
        print("loading dataset: ", i)
        set_features = np.load(cwd + "/database/ZINC/features/" + str(i) + ".npy")
        features = np.append(features,set_features, axis=0)
        set_adjs = np.load(cwd + "/database/ZINC/adj/" + str(i) + ".npy")
        adjs = np.append(adjs, set_adjs, axis=0)
    data = [features,adjs]
    target = logP[0:num_files*10000]
    history = model.fit(data, target,batch_size=32, epochs = epochs, validation_split=0.15)
    plot_training(history)
'''    
    for i in range(1):
        print("dataset: ", i+1)
        target = logP[i*10000:(i+1)*10000]
        print("targets in range: ",i*10000,"-",(i+1)*10000)
        print("Data files used: from /database/ZINC/features/" +str(i) + ".npy, from /database/ZINC/adj/" +str(i) + ".npy") 
        data = [np.load(cwd + "/database/ZINC/features/" + str(i) + ".npy"),np.load(cwd + "/database/ZINC/adj/" + str(i) + ".npy")]
        #print("From dataset:, " + str(i), "-",data[0])
        history = model.fit(data, target,batch_size=32, epochs = 10, validation_split=0.15)
'''

    
def plot_training(history):
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Model accuracy')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



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
    y = Flatten()(y)
    layers_list.append(y)
    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    
tf.keras.backend.clear_session()
cwd = os.getcwd()

adj = np.load(cwd + "/database/ZINC/adj/" + str(0) + ".npy")
feat = np.load(cwd + "/database/ZINC/features/" + str(0) + ".npy")
logP = np.load(cwd + "/database/ZINC/logP.npy")
target = logP[0:10000]
X = feat
A = adj
input_shapes = [Input(feat[0].shape),Input(adj[0].shape)]
Xinput = Input(feat[0].shape)
Ainput = Input(adj[0].shape)
output_dim = [50,32]
inputs = [Xinput,Ainput]


print("Which network will you use?")
print("1: Graph Convolution Network")
print("2: Graph Convolution Network + attention")
print("3: Graph Convolution Network + gated skip connection")
print("4: Graph Convolution Network + attention + gated skip connection")
#net_type = int(input("-- "))
#num_layers = input("Number of layers: ")

net_type=1
num_layers=1

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
