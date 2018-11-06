
from math import ceil

import numpy as np
import tensorflow as tf

class DropBNN(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        input_dim = np.prod(obs_shape) + np.prod(act_shape)
        input_tensor = Input(shape=input_dim)
        x = Dense(500, activation='relu')(input_tensor)
        x = Dropout(0.5)(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(0.5)(x)

        logits = Dense(output_classes)(x)
        variance_pre = Dense(1)(x)
        variance = Activation('softplus', name='variance')(variance_pre)
        logits_variance = concatenate([logits, variance], name='logits_variance')
        softmax_output = Activation('softmax', name='softmax_output')(logits)

        self.model = Model(inputs=input_tensor, outputs=[logits_variance,softmax_output])

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)

class EpistemicUncertaintyModel(object):
    def __init__(self, checkpoint, epistemic_monte_carlo_simulations):
        self.model = load_bayesian_model(checkpoint)
        inpt = Input(shape=(model.input_shape[1:]))
        x = RepeatVector(epistemic_monte_carlo_simulations)(inpt)
        # Keras TimeDistributed can only handle a single output from a model :(
        # and we technically only need the softmax outputs.
        hacked_model = Model(inputs=self.model.inputs, outputs=self.model.outputs[1])
        x = TimeDistributed(hacked_model, name='epistemic_monte_carlo')(x)
        # predictive probabilities for each class
        softmax_mean = TimeDistributedMean(name='epistemic_softmax_mean')(x)
        variance = PredictiveEntropy(name='epistemic_variance')(softmax_mean)
        epistemic_model = Model(inputs=inpt, outputs=[variance, softmax_mean])

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)
