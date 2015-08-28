import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne import PrintLog

import matplotlib.pyplot as plt

from nolearn.lasagne.visualize import plot_conv_weights

import gzip
import cPickle as pickle
import os

from skimage import io
from skimage import img_as_float

import numpy as np

import sys

import re

import nolearn_vis
import draw_net


net = NeuralNet(
    layers = [
        ('l_in', layers.InputLayer),
        ('l_conv1', layers.Conv2DLayer),
        ('l_pool1', layers.MaxPool2DLayer),
        ('l_conv2', layers.Conv2DLayer),
        ('l_pool2', layers.MaxPool2DLayer),
        ('l_hidden', layers.DenseLayer),
        ('l_out', layers.DenseLayer)
    ],

    l_in_shape = (None, 1, 28, 28),
    l_conv1_filter_size=(5,5), l_conv1_num_filters=20/2,
    l_pool1_pool_size=(2,2),
    l_conv2_filter_size=(5,5), l_conv2_num_filters=50/2,
    l_pool2_pool_size=(2,2),
    l_hidden_num_units=500/2, l_hidden_nonlinearity=lasagne.nonlinearities.rectify, l_hidden_W=lasagne.init.GlorotUniform(),
    l_out_num_units=10, l_out_nonlinearity=lasagne.nonlinearities.softmax, l_out_W=lasagne.init.GlorotUniform(),

    update=nesterov_momentum,
    update_learning_rate=0.001,
    update_momentum=0.9,
    verbose=1,
    max_epochs=10
)

"""
X_train = np.asarray([ nolearn_vis.load_image("1002_c2.gif") ], dtype="float32")
y_train = np.asarray([2], dtype="int32")
net.initialize()

print net.layers_

nolearn_vis.plot_network(net.layers_, X_train)

#net.fit(X_train, y_train)
"""

net.initialize_layers()

f = gzip.open("mnist.pkl.gz")
args = pickle.load(f)
f.close()

X_train = args["X_train"]
X_train = X_train.reshape( (10000, 1, 28, 28) )
y_train = np.asarray( args["y_train"].flatten(), dtype="int32" )

print dir(net.layers_)

layers = []
for elem in net.layers_:
    layers.append(net.layers_[elem])

print draw_net.draw_to_file(layers, "/tmp/graph.png")

net.fit(X_train, y_train)

#nolearn_vis.plot_loss(net)

nolearn_vis.plot_network(net.layers_, X_train[0:1])
