import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

#parameters
n_pixels = 28*28

X = tf.placeholder(tf.float32, shape=([None, n_pixels]))

#
def weight_variables(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def FC_layer(X, W, b) :
    return tf.matmul(X, W) + b

#latent representation
latent_dim = 20
h_dim = 500

#layer 1
W_enc = weight_variables([n_pixels, h_dim], 'W_enc')
b_enc = bias_variable([h_dim], 'b_enc')

#tanh - activation
h_enc = tf.nn.tanh(FC_layer(X, W_enc, b_enc))

#layer 2
W_mu = weight_variables([n_pixels, h_dim], 'W_mu')
b_mu = bias_variable([h_dim], 'b_mu')
mu = FC_layer(X, W_mu, b_mu) #mean

#standard deviation
W_logstd = weight_variables([h_dim, latent_dim], 'W_logstd')
b_logstd = bias_variable([latent_dim], 'b_logstd')
mu = FC_layer(h_enc, W_logstd, b_logstd) #std

#stop here