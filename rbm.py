#! coding=utf-8
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams


class RBM:
    def __init__(self, inputs=None, n_visiable=784, n_hidden=500, W=None, h_bias=None, v_bias=None, numpy_rng=None,
                 theano_rng=None):
        self.inputs = inputs
        self.n_visiable = n_visiable
        self.n_hidden = n_hidden
        if W is None:
            bound_value = np.sqrt(6.0 / (n_visiable + n_hidden))
            w_value = np.array(numpy_rng.uniform(low=- bound_value, high=bound_value, size=(n_visiable, n_hidden)), \
                               dtype=theano.config.floatX)
            self.W = theano.shared(value=w_value, name='W', borrow=True)
        else:
            self.W = W
        if v_bias is None:
            v_bias_val = np.zeros(n_visiable, dtype=theano.config.floatX)
            self.v_bias = theano.shared(value=v_bias_val, name='v_bias', borrow=True)
        else:
            self.v_bias = v_bias
        if h_bias is None:
            h_bias_val = np.zeros(n_hidden, dtype=theano.config.floatX)
            self.h_bias = theano.shared(value=h_bias_val, name='h_bias', borrow=True)
        else:
            self.h_bias = h_bias
        self.theata = theano.shared(value=np.ones(n_visiable, dtype=theano.config.floatX), name='theata', borrow=True)
        self.params = [self.W, self.h_bias, self.v_bias]
        if theano_rng is None:
            self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        else:
            self.theano_rng = theano_rng
        self.wx = T.dot(inputs, self.W)
        self.outputs = T.nnet.sigmoid(self.wx)
        self.b_term_v = T.dot(inputs, self.v_bias)
        self.b_term_h = T.dot(self.outputs, self.h_bias)

    def propup(self, vis):
        z = T.dot(vis, self.W) + self.h_bias
        activation = T.nnet.sigmoid(z)
        return [z, activation]

    def sample_h_given_v(self, vis):
        pre_activate_v, activate_v = self.propup(vis)
        h_sample = self.theano_rng.binomial(size=activate_v.shape, n=1, p=activate_v, dtype=theano.config.floatX)
        return [pre_activate_v, activate_v, h_sample]

    def propdown(self, hid):
        z = T.dot(hid, self.W.T) + self.v_bias
        activation = T.nnet.sigmoid(z)
        return [z, activation]

    def sample_v_given_h(self, hid):
        pre_activate_h, activate_h = self.propdown(hid)
        v_sample = self.theano_rng.binomial(size=activate_h.shape, n=1, p=activate_h, dtype=theano.config.floatX)
        return [pre_activate_h, activate_h, v_sample]

    def gibbs_hvh(self, hid):
        pre_activate_h, activate_h, v_sample = self.sample_v_given_h(hid)
        pre_activate_v, activate_v, h_sample = self.sample_h_given_v(v_sample)
        return [pre_activate_h, activate_h, v_sample, pre_activate_v, activate_v, h_sample]

    def gibbs_vhv(self, vis):
        pre_activate_v, activate_v, h_sample = self.sample_h_given_v(vis)
        pre_activate_h, activate_h, v_sample = self.sample_v_given_h(h_sample)
        return [pre_activate_v, activate_v, h_sample, pre_activate_h, activate_h, v_sample]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.h_bias
        # v_term = T.sum((v_sample - self.v_bias)**2/(2*self.theata**2), axis=1)
        v_term = T.dot(v_sample, self.v_bias)
        h_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -v_term - h_term

    def get_reconstruction(self, pre_activate):
        cross_entroy = -T.mean(T.sum(self.inputs * T.log(T.nnet.sigmoid(pre_activate)) + (1 - self.inputs) * T.log(
            1 - T.nnet.sigmoid(pre_activate)), axis=1))
        return cross_entroy

    def cost_updates(self, lr=0.01, persistent=None, k_step=1):
        # pre_actvivate_v, activate_v, h_samlpe = self.sample_h_given_v(self.inputs)
        if persistent is None:
            chain_start = self.inputs
        else:
            chain_start = persistent
        [pre_activate_v, activate_v, h_sample, pre_activate_h, activate_h, v_sample, ], updates = theano.scan(
            self.gibbs_vhv, outputs_info=[None, None, None, None, None, chain_start], n_steps=k_step)
        chain_end = v_sample[-1]
        cost = T.mean(self.free_energy(self.inputs)) - T.mean(self.free_energy(chain_end))
        grad_params = T.grad(cost, self.params, consider_constant=[chain_end])
        # updates = [(param, param - lr * g_param) for param, g_param in zip(self.params, grad_params)]
        # about cd-k gibbs sampling,the updates can be used to update the each loop in scan,if use the updates who \
        # calculated from grad, the scan will export the error ItermissError
        for gparam, param in zip(grad_params, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        if persistent is None:
            monitor_cost = self.get_reconstruction(pre_activate_h[-1])
        return monitor_cost, updates

    def get_hidden_feature(self, data):
        return self.propup(data)
