from keras.datasets import fashion_mnist
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import networkx as nx


def get_digit_data():
    # Load the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # Shuffle the training data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    # Shuffle the test data.
    permutation = np.random.permutation(x_test.shape[0])
    x_test = x_test[permutation]
    y_test = y_test[permutation]
    # Create the one-hot versions of y
    y_train_one_hot = np.zeros((x_train.shape[0], 10))
    y_test_one_hot = np.zeros((x_test.shape[0], 10))
    for i, y in enumerate(y_train):
        y_train_one_hot[i][y] = 1
    for i, y in enumerate(y_test):
        y_test_one_hot[i][y] = 1
    class_names = [str(i) for i in range(10)]
    return x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_names


# Create a hierarchical tree from origin to even/odd to respective digits.
# This reflects the true hierarchy of the data.
def get_parity_tree():
    tree = nx.DiGraph()
    for label in [0, 2, 4, 6, 8]:
        tree.add_edge('even', str(label))
        tree.add_edge('odd', str(label + 1))
    tree.add_edge('origin', 'even')
    tree.add_edge('origin', 'odd')
    nodes = list(tree.nodes)
    for node in nodes:
        tree.add_node(node, name=node)
    return tree


# Create the MNIST digit tree developed in metric-guided prototype learning.
# Untested.
def get_guided_tree():
    tree = nx.DiGraph()
    # First level of internal nodes
    tree.add_edge('layer1_dummy0', 1)
    tree.add_edge('layer1_dummy0', 7)
    tree.add_edge('layer1_dummy1', 0)
    tree.add_edge('layer1_dummy1', 6)
    tree.add_edge('layer1_dummy1', 9)
    tree.add_edge('layer1_dummy2', 8)
    tree.add_edge('layer1_dummy2', 3)
    tree.add_edge('layer1_dummy3', 5)
    tree.add_edge('layer1_dummy3', 2)
    # Second level
    tree.add_edge('layer2_dummy0', 'layer1_dummy0')
    tree.add_edge('layer2_dummy0', 4)
    tree.add_edge('layer2_dummy1', 'layer1_dummy1')
    tree.add_edge('layer2_dummy1', 'layer1_dummy2')
    # Tree root
    tree.add_edge('origin', 'layer2_dummy0')
    tree.add_edge('origin', 'layer2_dummy1')
    for node in list(tree.nodes):
        tree.add_node(node, name=node)
    return tree

# Label 	Description
# 0 	    T-shirt/top
# 1 	    Trouser
# 2 	    Pullover
# 3 	    Dress
# 4 	    Coat
# 5 	    Sandal
# 6 	    Shirt
# 7 	    Sneaker
# 8 	    Bag
# 9 	    Ankle boot
def get_fashion_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # Shuffle the training data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    # Shuffle the test data.
    permutation = np.random.permutation(x_test.shape[0])
    x_test = x_test[permutation]
    y_test = y_test[permutation]
    # Create the one-hot versions of y
    y_train_one_hot = np.zeros((x_train.shape[0], 10))
    y_test_one_hot = np.zeros((x_test.shape[0], 10))
    for i, y in enumerate(y_train):
        y_train_one_hot[i][y] = 1
    for i, y in enumerate(y_test):
        y_test_one_hot[i][y] = 1
    return x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_names


def get_fashion_tree():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    high_level = ['shoes', 'top', 'fancy']
    class_to_group_mapping = {0: 1,
                              1: 2,
                              2: 1,
                              3: 2,
                              4: 1,
                              5: 0,
                              6: 1,
                              7: 0,
                              8: 2,
                              9: 0}
    tree = nx.DiGraph()
    for low, high in class_to_group_mapping.items():
        tree.add_edge(high_level[high], class_names[low])
    for high in high_level:
        tree.add_edge('origin', high)
    nodes = list(tree.nodes)
    for node in nodes:
        tree.add_node(node, name=node)
    return tree


def make_noisy(x, noise_level=0.5):
    noise = np.random.normal(loc=0, scale=noise_level, size=x.shape)
    x_noisy = np.clip(x + noise, 0, 1)
    return x_noisy


# python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 21:22:58 2017
"""
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def get_conv_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    class_names = [i for i in range(10)]
    return x_train, y_train, to_categorical(y_train), x_test, y_test, to_categorical(y_test), class_names


def batch_elastic_transform(images, sigma, alpha, height, width, random_state=None):
    '''
    this code is borrowed from chsasank on GitHubGist
    Elastic deformation of images as described in [Simard 2003].

    images: a two-dimensional numpy array; we can think of it as a list of flattened images
    sigma: the real-valued variance of the gaussian kernel
    alpha: a real-value that is multiplied onto the displacement fields

    returns: an elastically distorted image of the same shape
    '''
    # print(images.shape)
    assert len(images.shape) == 2
    batch_size = images.shape[0]
    # the two lines below ensure we do not alter the array images
    e_images = np.empty_like(images)
    e_images[:] = images

    e_images = e_images.reshape((-1, height, width))
    if random_state is None:
        random_state = np.random.RandomState(None)
    x, y = np.mgrid[0:height, 0:width]

    for i in range(e_images.shape[0]):
        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        indices = x + dx, y + dy
        e_images[i] = map_coordinates(e_images[i], indices, order=1)

    return e_images.reshape(batch_size, -1)
