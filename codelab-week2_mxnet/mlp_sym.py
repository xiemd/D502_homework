# -*- coding: utf-8 -*
import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer, kernel=(3, 3), num_filter=64, activation='relu', pool_stride=(2, 2), pool=False, BN=False):
    """
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    l = mx.sym.Convolution(data=input_layer, kernel=kernel, num_filter=num_filter, pad=(1, 1))
    if BN:
        l = mx.sym.BatchNorm(l)
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if pool:
        l = mx.sym.Pooling(data=l, pool_type="max", stride=pool_stride, kernel=(2, 2))
    return l
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?




# Optional
def inception_layer():
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer我们的攻城狮正在努力修复这个bug中
    """
    pass


def get_conv_sym():

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    conv1 = conv_layer(data, num_filter=128, pool=True, BN=True)
    conv2 = conv_layer(conv1, num_filter=64, pool=True, BN=True)
    flat = mx.sym.flatten(data=conv2)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=100)
    ac1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=10)
    # softmax loss
    l = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

    return l



    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?
