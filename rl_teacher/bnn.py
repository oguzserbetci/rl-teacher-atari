#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CODE FROM: https://github.com/thu-ml/zhusuan/blob/master/examples/bayesian_neural_nets/bayesian_nn.py

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs

import dataset

class BayesianNN(object):
    def __init__(self, obs_shape, act_shape, observed, n_particles, segment_length, batchsize, h_size=64):
        self.n_particles = n_particles
        self.batchsize = batchsize
        self.segment_length = segment_length

        with zs.BayesianNet(observed=observed) as model:
            input_dim = np.prod(obs_shape) + np.prod(act_shape)
            layer_sizes = [input_dim] + [64] + [1]
            self.w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

            self.ws = []
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                w_mu = tf.zeros([1, n_out, n_in + 1])
                self.ws.append(zs.Normal('w' + str(i),
                                         w_mu,
                                         std=1.,
                                         n_samples=n_particles, group_ndims=2))

            self.model = model

    def run(self, obs, acts):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, acts], axis=1)

        # forward
        # replicate input to sample many networks
        ly_x = tf.expand_dims(tf.tile(tf.expand_dims(x, 0), [self.n_particles, 1, 1]), 3)
        for i in range(len(self.ws)):
            # tile weights per batch and frame
            w = tf.tile(self.ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat([ly_x, tf.ones([self.n_particles, tf.shape(x)[0], 1, 1])], 2)
            # forward pass
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.to_float(tf.shape(ly_x)[2]))
            # add relu activation if not last layer
            if i < len(self.ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        # y_mean = fNN(x, W)
        reward_mean = tf.squeeze(ly_x, [2, 3])

        # reshape rewards to sum up segments
        segment_rewards = tf.reshape(reward_mean, (-1, self.batchsize, self.segment_length))
        segment_rewards = tf.reduce_sum(segment_rewards, axis=2)

        # y ~ N(y|y_mean, y_logstd)  : noise is added to the output to get a tractable likelihood
        segment_logstd = tf.get_variable('segment_logstd', shape=[],
                                         initializer=tf.constant_initializer(0.))
        _ = zs.Normal('segment_rewards', segment_rewards, logstd=segment_logstd)

        return reward_mean, segment_rewards


@zs.reuse('model')
def bayesianNN(observed, x, n_x, layer_sizes, n_particles, batchsize, segment_length):
    with zs.BayesianNet(observed=observed) as model:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mu = tf.zeros([1, n_out, n_in + 1])
            ws.append(
                zs.Normal('w' + str(i), w_mu, std=1.,
                          n_samples=n_particles, group_ndims=2))

        # forward
        # replicate input to sample many networks
        ly_x = tf.expand_dims(tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
        for i in range(len(ws)):
            # tile weights per batch and frame
            w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
            ly_x = tf.concat([ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
            # forward pass
            ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.to_float(tf.shape(ly_x)[2]))
            # add relu activation if not last layer
            if i < len(ws) - 1:
                ly_x = tf.nn.relu(ly_x)

        # y_mean = fNN(x, W)
        reward_mean = tf.squeeze(ly_x, [2, 3])

        # reshape rewards to sum up segments
        segment_rewards = tf.reshape(reward_mean, (-1, batchsize, segment_length))
        segment_rewards = tf.reduce_sum(segment_rewards, axis=2)

        # y ~ N(y|y_mean, y_logstd)  : noise is added to the output to get a tractable likelihood
        segment_logstd = tf.get_variable('segment_logstd', shape=[],
                                   initializer=tf.constant_initializer(0.))
        _ = zs.Normal('segment_rewards', segment_rewards, logstd=segment_logstd)

    return model, reward_mean, None, segment_rewards


def mean_field_variational(layer_sizes, n_particles):
    with zs.BayesianNet() as variational:
        ws = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                              layer_sizes[1:])):
            w_mean = tf.get_variable(
                'w_mean_' + str(i), shape=[1, n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            w_logstd = tf.get_variable(
                'w_logstd_' + str(i), shape=[1, n_out, n_in + 1],
                initializer=tf.constant_initializer(0.))
            ws.append(
                zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                          n_samples=n_particles, group_ndims=2))
    return variational


def main():
    tf.set_random_seed(1237)
    np.random.seed(1234)

    # Load UCI Boston housing data
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
        dataset.load_uci_boston_housing('./housing')
    x_train = np.vstack([x_train, x_valid])
    y_train = np.hstack([y_train, y_valid])
    N, n_x = x_train.shape

    # Standardize data
    x_train, x_test, _, _ = dataset.standardize(x_train, x_test)
    y_train, y_test, mean_y_train, std_y_train = dataset.standardize(
        y_train, y_test)

    # Define model parameters
    n_hiddens = [50]

    print(x_train.shape)
    print(y_train.shape)
    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x = tf.placeholder(tf.float32, shape=[None, n_x])
    y = tf.placeholder(tf.float32, shape=[None])
    layer_sizes = [n_x] + n_hiddens + [1]
    w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

    def log_joint(observed):
        model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
        log_pws = model.local_log_prob(w_names)
        log_py_xw = model.local_log_prob('y')
        return tf.add_n(log_pws) + log_py_xw * N

    variational = mean_field_variational(layer_sizes, n_particles)
    qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
    latent = dict(zip(w_names, qw_outputs))
    lower_bound = zs.variational.elbo(
        log_joint, observed={'y': y}, latent=latent, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    infer_op = optimizer.minimize(cost)

    # prediction: rmse & log likelihood
    observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
    observed.update({'y': y})
    model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
    y_pred = tf.reduce_mean(y_mean, 0)
    rmse = tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train
    log_py_xw = model.local_log_prob('y')
    log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - \
        tf.log(std_y_train)

    # Define training/evaluation parameters
    lb_samples = 10
    ll_samples = 5000
    epochs = 500
    batch_size = 10
    iters = int(np.floor(x_train.shape[0] / float(batch_size)))
    test_freq = 10

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]

                qwo, lb = sess.run([qw_outputs,lower_bound ],
                    feed_dict={n_particles: lb_samples,
                               x: x_batch, y: y_batch})

                _, lb = sess.run(
                    [infer_op, lower_bound],
                    feed_dict={n_particles: lb_samples,
                               x: x_batch, y: y_batch})
                lbs.append(lb)
            print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                test_lb, test_rmse, test_ll = sess.run(
                    [lower_bound, rmse, log_likelihood],
                    feed_dict={n_particles: ll_samples,
                               x: x_test, y: y_test})
                print('>> TEST')
                print('>> Test lower bound = {}, rmse = {}, log_likelihood = {}'
                      .format(test_lb, test_rmse, test_ll))


if __name__ == "__main__":
    main()
