#! conding=utf-8
# author = sai# -*- coding: utf-8 -*-
# author = sai
import theano, rbm, logistic
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams


class DBM:
    def __init__(self, numpt_rng, theano_rng=None, n_in=784, hidden_layers_size=[500, 100], n_out=10):
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_size)
        self.current_index = 0
        assert self.n_layers >= 0
        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpt_rng.randint(2 ** 30))
        self.theano_rng = theano_rng
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layers_size[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.rbm_layers[-1].outputs
            rbm_layer = rbm.RBM(
                inputs=layer_input,
                n_visiable=input_size,
                n_hidden=hidden_layers_size[i],
                numpy_rng=numpt_rng,
                theano_rng=theano_rng,
            )
            self.params.extend(rbm_layer.params)
            self.rbm_layers.append(rbm_layer)
    # index表示当前rbm的索引
    def propup(self, current_inputs):
        return self.rbm_layers[self.current_index].propup(current_inputs)

    def sample_h_given_v(self, vis):
        return self.rbm_layers[self.current_index].sample_h_given_v(vis)

    def propdown(self, current_inputs):
        return self.rbm_layers[self.current_index].propdown(current_inputs)

    def sample_v_given_h(self, vis):
        return self.rbm_layers[self.current_index].sample_v_given_h(vis)

    def gibbs_vhv(self, current_inputs):
        return self.rbm_layers[self.current_index].gibbs_vhv(current_inputs)

    def gibbs_hvh(self, hid):
        return self.rbm_layers[self.current_index].gibbs_hvh(hid)

    # 以上用于端层得采样

    def gibbs_curr2curr(self, current_inputs):
        activation_up, z1 = self.propup(current_inputs)
        activation_up_down, z2 = self.propdown(z1)
        index = self.current_index
        self.current_index -= 1
        activation_down, z3 = self.propdown(current_inputs)
        activation_down_up, z4 = self.propup(z3)
        self.current_index = index
        activation = T.dot(activation_up_down, self.rbm_layers[self.current_index].W.T) + \
                     T.dot(activation_down_up, self.rbm_layers[self.current_index - 1].W) + self.rbm_layers[self.current_index].v_bias
        z = T.nnet.sigmoid(activation_down_up)
        curr_sample = self.theano_rng.binomial(size=activation.shape, n=1, p=z, dtype=theano.config.floatX)
        return [None, None, None, activation, z, curr_sample]

    def log_dot(self, acc):
        return T.sum(T.log(1 + T.exp(acc)), axis=1)

    def free_energy(self, current_inputs, index):
        print 'calculate the free energy..'
        '''
            对于能量函数，三种类型：最底层，最上层，中间层
            分别为当前偏执项、其余层激活似然值得和
        '''
        energy = theano.shared(value=np.zeros(1,dtype=theano.config.floatX), name='energy', borrow=True)
        if index == 0:
            self.current_index = index
            b_term = self.rbm_layers[self.current_index].b_term_v
            temp = self.current_index
            while self.current_index < len(self.rbm_layers) - 1:
                [activation, z] = self.propup(current_inputs)
                energy += self.log_dot(activation)
                current_inputs = z
                self.current_index += 1
            self.current_index = index
            return -energy - b_term
        elif index == len(self.rbm_layers) - 1:
            self.current_index = index
            b_term = self.rbm_layers[self.current_index].b_term_h
            while self.current_index >= 0:
                [activation, z] = self.propdown(current_inputs)
                energy += self.log_dot(activation)
                current_inputs = z
                self.current_index -= 1
            self.current_index = index
            return -energy - b_term
        else:
            print self.current_index, index
            self.current_index = index
            # 做修改
            b_term = self.rbm_layers[index].b_term_v
            while self.current_index < len(self.rbm_layers) - 1:
                [activation, z] = self.propup(current_inputs)
                energy += self.log_dot(activation)
                current_inputs = z
                self.current_index += 1
            self.current_index = index
            while self.current_index >= 0:
                [activation, z] = self.propdown(current_inputs)
                energy += self.log_dot(activation)
                current_inputs = z
                self.current_index -= 1
            self.current_index = index
            return -energy - b_term

    def get_reconstruction(self, current_inputs, activation):
        cross_entroy = -T.mean(
            T.sum(current_inputs * T.log(activation) + (1 - current_inputs) * T.log(1 - activation), axis=1))
        return cross_entroy

    def sample_cost(self, chain_start, chain_end, updates, index, lr):
        cost = T.mean(self.free_energy(chain_start, index)) - T.mean(self.free_energy(chain_end, index))
        if 0<self.current_index<len(self.rbm_layers)-1:
            params = self.params
        else:
            params = self.rbm_layers[self.current_index].params
        grad_params = T.grad(cost, params, consider_constant=[chain_end])
        for gparam, param in zip(grad_params, params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        return updates
    def loop(self, current_inputs, fn, index, k_step, lr):
        chain_start = current_inputs
        [pre_activate_v, activate_v, h_sample, pre_activate_h, activate_h, v_sample], updates \
            = theano.scan(fn, outputs_info=[None, None, None, None, None, chain_start],
                          n_steps=k_step)
        chain_end = v_sample[-1]
        updates = self.sample_cost(chain_start, chain_end, updates, index, lr)
        cross_entroy = self.get_reconstruction(current_inputs, activate_h[-1])
        return cross_entroy, updates

    def cost_update(self, index, lr=0.01, k_step=1):
        if index == 0:
            self.current_index = index
            inputs = self.rbm_layers[self.current_index].inputs
            cross_entroy, updates = self.loop(inputs, self.gibbs_vhv, index, k_step, lr)
            return cross_entroy, updates
        elif index == len(self.rbm_layers) - 1:
            self.current_index = index
            inputs = self.rbm_layers[self.current_index].outputs
            cross_entroy, updates = self.loop(inputs, self.gibbs_hvh, index, k_step, lr)
            return cross_entroy, updates
        else:
            self.current_index = index
            inputs = self.rbm_layers[self.current_index].inputs
            cross_entroy, updates = self.loop(inputs, self.gibbs_curr2curr, index, k_step, lr)
            return cross_entroy, updates

    def pretrainging_function(self, train_set_x, batch_size, k):
        index = T.iscalar('index')
        learning_rate = T.scalar('lr')
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        pretrain_fns = []

        for ind in xrange(len(self.rbm_layers)):
            cost, updates = self.cost_update(ind, learning_rate, k_step=k)
            fn = theano.function(
                inputs=[index, learning_rate],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)
        return pretrain_fns
