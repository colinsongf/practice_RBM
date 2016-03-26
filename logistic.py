# -*- coding: utf-8 -*-
# author = sai

import theano
import theano.tensor as T
import numpy as np


class Softmax_layer:
    def __init__(self, inputs, n_in, n_out):
        rng = np.random.RandomState(1234)
        w_val = np.array(rng.uniform(np.sqrt(4. / (n_in + n_out)), np.sqrt(4. / (n_out + n_in)), size=(n_in, n_out)),
                         theano.config.floatX)
        self.W = theano.shared(value=w_val, name='w', borrow=True)
        b_val = np.zeros(n_out, theano.config.floatX)
        self.b = theano.shared(value=b_val, name='b', borrow=True)
        self.params = [self.W, self.b]
        self.outputs = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        self.pred_y = T.argmax(self.outputs, axis=1)

    def nagetive_likehood(self, y):
        return -T.mean(T.log(self.outputs)[T.arange(y.shape[0]), y])

    def error(self, y):
        return T.mean(T.neq(y, self.pred_y))


# this hidden layer is youdu
class Hidden_layer:
    def __init__(self, inputs, n_in, n_out):
        rng = np.random.RandomState(123)
        w_val = np.array(rng.uniform(np.sqrt(4. / (n_in + n_out)), np.sqrt(4. / (n_out + n_in)), size=(n_in, n_out)),
                         theano.config.floatX)
        self.W = theano.shared(value=w_val, name='w_h', borrow=True)
        b_val = np.zeros(n_out, theano.config.floatX)
        self.b = theano.shared(value=b_val, name='b_h', borrow=True)
        self.params = [self.W, self.b]
        self.outputs = T.nnet.sigmoid(T.dot(inputs, self.W) + self.b)
        self.wx = T.dot(self.outputs, self.W.T)
        self.b_term = T.dot(self.outputs, self.b)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
        else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]