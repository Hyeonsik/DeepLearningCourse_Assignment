# This library is used for Assignment3_Part2_ImageCaptioning

# Write your own image captiong code
# You can modify the class structure
# and add additional function needed for image captionging

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.nn import rnn_cell
import functools, sys

def lazy_property(function, name = None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    #name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Captioning():
    def __init__(self, batch_size):
        self._start = 1
        self.hidden_size = 512 # length of hidden units (input features length is 512)
        self.timesteps = 17 # length of time steps for every sample 
        self.n_words = 1004
        self.batch_size = batch_size
        self.h0 = tf.placeholder(tf.float32, shape = [batch_size, self.hidden_size])
        self.labels = tf.placeholder(tf.float32, shape = [batch_size, self.timesteps, self.n_words])
        self.keep_probs = tf.placeholder(tf.float32)

        self.build_model
        self.train

    @lazy_property
    def build_model(self):
        lstm = tf.contrib.rnn.LSTMBlockCell(self.hidden_size)
        #add x0 = [0,0 ... ,0]
        x0 = tf.zeros([self.batch_size, 1, self.n_words])
        _input = tf.concat([x0, self.labels[:,:self.timesteps-1,:]], axis = 1)

        #initial state of LSTM
        c_init = np.zeros((self.batch_size, lstm.state_size.c), np.float32)
        h_init = self.h0
        state_in = tf.contrib.rnn.LSTMStateTuple(c_init, h_init)

        hidden_output, _= tf.nn.dynamic_rnn(lstm, _input, initial_state = state_in, time_major = False, dtype = tf.float32)

        Wy = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_words])) # hidden state : (batch_size, hidden_len=512) / y : (batch_size, n_word = 1004)
        by = tf.Variable(tf.zeros([self.n_words]), dtype = tf.float32)  

        hidden_unstack = tf.unstack(hidden_output, self.timesteps, axis=1)
        self.labels_unstack = tf.unstack(self.labels, self.timesteps, axis=1)

        losses = 0
        accuracy = 0
        self.outputs = []
        for i in range(self.timesteps):
            output = tf.matmul(hidden_unstack[i], Wy) + by
            output = tf.nn.tanh(output)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = self.labels_unstack[i]))
            losses += loss
            self.outputs.append(output)
            #accuracy += tf.reduce_sum(tf.math.equal(tf.argmax(output,axis=1), tf.argmax(labels_us[i], axis = 1)))
        losses /= self.timesteps
        accuracy /= self.timesteps
        self.loss = losses
        return self.loss


    @lazy_property
    def train(self):
        self.optimize = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        #self.optimize = tf.GradientDescentOptimizer(learning_rate = 0.1).minimize(self.loss)
        return self.optimize

	#@lazy_property
    def predict(self):
        captions = None
        correct = tf.equal(tf.argmax(self.prediction,1),tf.argmax(self.labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		

