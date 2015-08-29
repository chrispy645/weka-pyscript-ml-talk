import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax

from nolearn.lasagne import NeuralNet

import matplotlib.pyplot as plt

from nolearn.lasagne.visualize import plot_conv_weights

import gzip
import cPickle as pickle
import os

import numpy as np

import sys

import re

import nolearn_vis
import draw_net

def get_net(num_epochs):
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
        max_epochs=num_epochs,
    )
    return net

def train(args):
    X_train = args["X_train"]
    X_train = X_train.reshape( (X_train.shape[0], 1, 28, 28) )
    y_train = np.asarray( args["y_train"].flatten(), dtype="int32" )
    num_epochs = 1 if "num_epochs" not in args else args["num_epochs"]
    net = get_net(num_epochs)
    net.fit(X_train, y_train)
    if "vis" in args and args["vis"] == True:
        nolearn_vis.plot_conv_activity( net.layers_[1], X_train[0:1] )
    return net.get_all_params_values()

def describe(args, model):
    return "my conv net"

def test(args, model):
    X_test = args["X_test"]
    X_test = X_test.reshape( (X_test.shape[0], 1, 28, 28) )
    num_epochs = 1 if "num_epochs" not in args else args["num_epochs"]
    net = get_net(num_epochs)
    net.initialize()
    net.load_params_from(model)
    return net.predict_proba(X_test).tolist()

if __name__ == '__main__':

    #net.initialize_layers()

    f = gzip.open("mnist.pkl.gz")
    args = pickle.load(f)
    f.close()

    #args["vis"] = True
    args["num_epochs"] = 20
    model_params = train(args)

    args["X_test"] = args["X_train"]

    predictions = test(args, model_params)

    predicted_labels = []
    actual_labels = args["y_train"].flatten().tolist()
    for pred in predictions:
        predicted_labels.append( np.argmax(pred) )

    print "Accuracy is: %f" % (float( np.sum( np.equal(actual_labels, predicted_labels) ) ) / len(actual_labels))

    """
    layers = []
    for elem in net.layers_:
        layers.append(net.layers_[elem])

    #print draw_net.draw_to_file(layers, "/tmp/graph.png")

    #net.fit(X_train, y_train)

    #nolearn_vis.plot_loss(net)

    nolearn_vis.plot_network(net.layers_, X_train[0:1])
    """
