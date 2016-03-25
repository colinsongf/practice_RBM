# -*- coding: utf-8 -*-
# author = sai
import theano
import theano.printing as pt
import theano.tensor as T
import rbm
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

'''
    将特殊2层RBM修改至多层
'''
# global namespace
dtype = theano.config.floatX
shared = theano.shared


class DBM:
    def __init__(self, inputs=None, n_unit=[784, 1000, 784], numpy_rng=None, theano_rng=None):
        self.inputs = inputs
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        self.numpy_rng = numpy_rng
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(1234))
            self.theano_rng = theano_rng
        # 初始化参数
        sizes = [(n_unit[i], n_unit[i + 1]) for i in xrange(len(n_unit) - 1)]
        W_values = [np.array(self.numpy_rng.normal(size=size)) for size in sizes]
        b_values = [np.zeros(size) for size in n_unit]
        w_name = ['w' + str(i) for i in xrange(len(n_unit) - 1)]
        b_name = ['b' + str(i) for i in xrange(len(n_unit))]
        layers_name = ['layer' + str(i + 1) for i in xrange(len(n_unit))]
        self.W = [shared(w_value, name=name, borrow=True) for w_value, name in zip(W_values, w_name)]
        self.b = [shared(b_value, name=name, borrow=True) for b_value, name in zip(b_values, b_name)]
        self.params = self.W + self.b
        self.layers_values = [shared(value=numpy_rng.binomial(1, 0.5, size=[size]), name=name) for size, name in
                              zip(n_unit, layers_name)]

    def propup(self, current_inputs, index):
        if index > len(self.W) - 1:
            print 'propup时查出索引范围'
        activation = T.dot(current_inputs, self.W[index]) + self.b[index + 1]
        z = T.nnet.sigmoid(activation)
        return [activation, z]

    def propdown(self, current_inputs, index):
        if index - 1 < 0:
            print 'propdown时超出索引范围'
        activation = T.dot(current_inputs, self.W[index - 1].T) + self.b[index - 1]
        z = T.nnet.sigmoid(activation)
        return [activation, z]

    def sample_post_given_current(self, current_inputs, index):
        pre_activation, activation_up = self.propup(current_inputs, index)
        up_sample = self.theano_rng.binomial(size=activation_up.shape, n=1, p=activation_up, dtype=dtype)
        return [activation_up, up_sample]

    def sample_pre_given_current(self, current_inputs, index):
        pre_activation, activation_down = self.propdown(current_inputs, index)
        down_sample = self.theano_rng.binomial(size=activation_down.shape, n=1, p=activation_down)
        return [activation_down, down_sample]

    def fusion_inputs(self, down_inputs, up_inputs, index):
        '''
            index为当前层得索引，从0开始
            down_inputs, up_inputs分别为当前层propdown, propdown之后得到得值
        '''
        pre_activation = T.dot(down_inputs, self.W[index-1]) + T.dot(up_inputs, self.W[index+1].T) + self.b[index]
        activation_fusion = T.nnet.sigmoid(pre_activation)
        sample_up_down = self.theano_rng.binomial(size=activation_fusion.shape, n=1, p=activation_fusion, dtype=dtype)
        return [activation_fusion, sample_up_down]

    def gibbs_sample_curr2curr(self, current_inputs, index):
        '''
            三种采样方式：
                1.如果当前层为第一层，则先propup再propdown
                2.如果当前层为中间层，则需同时向邻近层propup和propdown，进而将邻近层激活值合并输入当前层
                3.如果当前层为最后一层，则先propdown再proup
        '''
        if index - 1 < 0:
            # down_sample采样后得到的当前层样本分布，当前层为最后一层
            activation_up, up_sample = self.sample_post_given_current(current_inputs, index)
            activation_down, down_sample = self.sample_pre_given_current(up_sample, index+1)
            return [activation_up, up_sample, activation_down, down_sample]
        elif index + 1 > len(self.layers_values):
            # up_sample为采样后得到得当前层样本分布，当前层为最后一层
            activation_down, down_sample = self.sample_pre_given_current(current_inputs, index)
            activation_up, up_sample = self.sample_post_given_current(down_sample, index-1)
            return [activation_down, down_sample, activation_up, up_sample]
        else:
            # up_down_sample为采样后得到的当前层样本分布，当前层为中间层
            down_curr_sample = self.sample_pre_given_current(current_inputs, index)
            up_curr_sample = self.sample_post_given_current(current_inputs, index)
            activation_fusion, up_down_sample = self.fusion_inputs(down_curr_sample, up_curr_sample)
            return [down_curr_sample, up_curr_sample, activation_fusion, up_down_sample]

    def get_layer(self, index):
        pass
    def free_energy(self, current):
        pass

    def cost_updates(self):
        pass

    def cross_entroy(self):
        pass
