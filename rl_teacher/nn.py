from math import ceil

import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.contrib import distributions
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model, load_model
from keras.layers import Layer, Input

# Bayesian categorical cross entropy.
# N data points, C classes, T monte carlo simulations
# true - true values. Shape: (N, C)
# pred_var - predicted logit values and variance. Shape: (N, C + 1)
# returns - loss (N,)


def bayesian_categorical_crossentropy(T, num_classes):
    def bayesian_categorical_crossentropy_internal(true, pred_var):
        # shape: (N,)
        std = K.sqrt(pred_var[:, num_classes:])
        # shape: (N,)
        variance = pred_var[:, num_classes]
        variance_depressor = K.exp(variance) - K.ones_like(variance)
        # shape: (N, C)
        pred = pred_var[:, 0:num_classes]
        # shape: (N,)
        undistorted_loss = K.categorical_crossentropy(
            pred, true, from_logits=True)
        # shape: (T,)
        iterable = K.variable(np.ones(T))
        dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
        monte_carlo_results = K.map_fn(
            gaussian_categorical_crossentropy(true, pred, dist,
                                              undistorted_loss, num_classes),
            iterable,
            name='monte_carlo_results')

        variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss

        return variance_loss + undistorted_loss + variance_depressor

    return bayesian_categorical_crossentropy_internal


def load_bayesian_model(checkpoint, monte_carlo_simulations=100, classes=10):
    get_custom_objects().update({
        "bayesian_categorical_crossentropy_internal":
        bayesian_categorical_crossentropy(monte_carlo_simulations, classes)
    })
    return load_model(checkpoint)


class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        # Import Keras here to avoid messing with multiprocessing context too early
        from keras.layers import Dense, Dropout, LeakyReLU
        from keras.models import Sequential

        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)


class SimpleConvolveObservationQNet(FullyConnectedMLP):
    """
    Network that has two convolution steps on the observation space before flattening,
    concatinating the action and being an MLP.
    """

    def __init__(self, obs_shape, act_shape, h_size=64):
        after_convolve_shape = (int(ceil(ceil(obs_shape[0] / 4) / 3)),
                                int(ceil(ceil(obs_shape[1] / 4) / 3)), 8)
        super().__init__(after_convolve_shape, act_shape, h_size)

    def run(self, obs, act):
        if len(obs.shape) == 3:
            # Need to add channels
            obs = tf.expand_dims(obs, axis=-1)
        # Parameters taken from GA3C NetworkVP
        c1 = tf.layers.conv2d(
            obs,
            4,
            kernel_size=8,
            strides=4,
            padding="same",
            activation=tf.nn.relu)
        c2 = tf.layers.conv2d(
            c1,
            8,
            kernel_size=6,
            strides=3,
            padding="same",
            activation=tf.nn.relu)
        return super().run(c2, act)


class BayesianModel(object):
    def __init__(self, obs_shape, act_shape, h_size=64):
        # Import Keras here to avoid messing with multiprocessing context too early
        from keras.layers import Dense, Dropout, LeakyReLU, Activation
        from keras.models import Sequential
        from keras.layers.merge import concatenate

        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        inpt = Input(shape=(input_dim,))
        x = Dense(h_size, input_dim=input_dim)(inpt)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)
        x = Dense(h_size)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.5)(x)

        reward = Dense(1)(x)
        variance_pre = Dense(1)(reward)
        variance = Activation('softplus', name='variance')(variance_pre)

        self.model = Model(
            inputs=inpt, outputs=[variance, reward])

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)


class EpistemicModel(object):
    def __init__(self, checkpoint, epistemic_monte_carlo_simulations):
        # Import Keras here to avoid messing with multiprocessing context too early
        from keras.layers import Dense, Dropout, RepeatVector, LeakyReLU, Activation
        from keras.models import Sequential
        from keras.layers.merge import concatenate
        from keras.layers.wrappers import TimeDistributed

        model = load_bayesian_model(checkpoint)
        inpt = Input(shape=(model.input_shape[1:]))
        x = RepeatVector(epistemic_monte_carlo_simulations)(inpt)
        # Keras TimeDistributed can only handle a single output from a model :(
        # and we technically only need the softmax outputs.
        hacked_model = Model(inputs=model.inputs, outputs=model.outputs[1])
        x = TimeDistributed(hacked_model, name='epistemic_monte_carlo')(x)
        # predictive probabilties for each class
        softmax_mean = TimeDistributedMean(name='epistemic_softmax_mean')(x)
        variance = PredictiveEntropy(name='epistemic_variance')(softmax_mean)
        self.model = Model(inputs=inpt, outputs=[variance, softmax_mean])

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)


# Apply the predictive entropy function for input with C classes.
# Input of shape (None, C, ...) returns output with shape (None, ...)
# Input should be predictive means for the C classes.
# In the case of a single classification, output will be (None,).
class PredictiveEntropy(Layer):
    def build(self, input_shape):
        super(PredictiveEntropy, self).build(input_shape)

    # input shape (None, C, ...)
    # output shape (None, ...)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], )

    # x - prediction probability for each class(C)
    def call(self, x):
        return -1 * K.sum(K.log(x) * x, axis=1)
