import os
import random
from time import sleep

from multiprocessing import Process

import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy import stats

from rl_teacher.summaries import AgentLogger
from rl_teacher.clip_manager import SynthClipManager, ClipManager
from rl_teacher.nn import FullyConnectedMLP, SimpleConvolveObservationQNet
from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.segment_sampling import segments_from_rand_rollout, sample_segment_from_path, basic_segment_from_null_action
from rl_teacher.utils import corrcoef

def nn_predict_rewards(obs_segments, act_segments, network, obs_shape, act_shape):
    """
    :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
    :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
    :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
    :param obs_shape: a tuple representing the shape of the observation space
    :param act_shape: a tuple representing the shape of the action space
    :return: tensor with shape = (batch_size, segment_length)
    """
    batchsize = tf.shape(obs_segments)[0]
    segment_length = tf.shape(obs_segments)[1]

    # Temporarily chop up segments into individual observations and actions
    obs = tf.reshape(obs_segments, (-1,) + obs_shape)
    acts = tf.reshape(act_segments, (-1,) + act_shape)

    # Run them through our neural network
    rewards = network.run(obs, acts)

    # Group the rewards back into their segments
    return tf.reshape(rewards, (batchsize, segment_length))

class RewardModel(object):
    is_fresh = False  # Set this to true to generate pretraining data by default.

    def predict_reward(self, path):
        raise NotImplementedError()  # Must be overridden

    def path_callback(self, path):
        pass  # Defaults to no behavior

    def reset_training_data(self, env_id, n_pretrain_labels, clip_length, stacked_frames, workers):
        pass  # No training data by default

    def train(self, iterations=1, report_frequency=None):
        pass  # Doesn't require training by default

    def save_model_checkpoint(self):
        pass  # Nothing to save

    def load_model_from_checkpoint(self):
        raise NotImplementedError()

class OriginalEnvironmentReward(RewardModel):
    """Model that always gives the reward provided by the environment."""

    def predict_reward(self, path):
        return path["original_rewards"]

class OrdinalRewardModel(RewardModel):
    """A learned model of an environmental reward using training data that is merely sorted."""

    def __init__(self, model_type, env, experiment_name, label_schedule, clip_length, stacked_frames, workers):
        model_type = "human"  # HACK TODO INTEGRATE
        if model_type == "synth":
            self.clip_manager = SynthClipManager(env, experiment_name)
        elif model_type == "human":
            self.clip_manager = ClipManager(env, experiment_name, workers)
        else:
            raise ValueError("Cannot find clip manager that matches keyword \"%s\"" % model_type)
        self.is_fresh = self.clip_manager.total_number_of_clips == 0

        self.label_schedule = label_schedule
        self.experiment_name = experiment_name
        self._frames_per_segment = clip_length * env.fps
        # The reward distribution has standard dev such that each frame of a clip has expected reward 1
        self._standard_deviation = self._frames_per_segment
        self._elapsed_training_iters = 0
        self._episode_count = 0
        self._episodes_per_training = 50
        self._iterations_per_training = 50
        self._episodes_per_checkpoint = 100

        # Build and initialize our model
        config = tf.ConfigProto(
            # device_count={'GPU': 0},
            # log_device_placement=True,
        )
        config.gpu_options.per_process_gpu_memory_fraction = 0.35  # allow_growth = True
        self.sess = tf.Session(config=config)

        self.obs_shape = env.observation_space.shape
        if stacked_frames > 0:
            self.obs_shape = self.obs_shape + (stacked_frames,)
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape

        self.graph = self._build_model()
        self.sess.run(tf.global_variables_initializer())
        my_vars = tf.global_variables()
        self.saver = tf.train.Saver({var.name: var for var in my_vars}, max_to_keep=0)

    def _build_model(self):
        """Our model takes in path segments with observations and actions, and generates rewards (Q-values)."""
        # Set up observation placeholder
        self.obs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None) + self.obs_shape, name="obs_placeholder")

        # Set up action placeholder
        if self.discrete_action_space:
            self.act_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None), name="act_placeholder")
            # Discrete actions need to become one-hot vectors for the model
            segment_act = tf.one_hot(tf.cast(self.act_placeholder, tf.int32), self.act_shape[0])
        else:
            self.act_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None) + self.act_shape, name="act_placeholder")
            # Assume the actions are how we want them
            segment_act = self.segment_act_placeholder

        # A neural network maps a (state, action) pair to a reward
        net = SimpleConvolveObservationQNet(self.obs_shape, self.act_shape)
        self.rewards = nn_predict_rewards(self.obs_placeholder, segment_act, net, self.obs_shape, self.act_shape)

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        self.segment_rewards = tf.reduce_sum(self.rewards, axis=1)

        self.targets = tf.placeholder(dtype=tf.float32, shape=(None,), name="reward_targets")

        self.loss = tf.reduce_mean(tf.square(self.targets - self.segment_rewards))

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        return tf.get_default_graph()

    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            predicted_rewards = self.sess.run(self.rewards, feed_dict={
                self.obs_placeholder: np.asarray([path["obs"]]),
                self.act_placeholder: np.asarray([path["actions"]]),
                K.learning_phase(): False
            })
        return predicted_rewards[0]  # The zero here is to get the single returned path.

    def path_callback(self, path):
        self._episode_count += 1

        # We may be in a new part of the environment, so we take a clip to learn from if requested
        if self.clip_manager.total_number_of_clips < self.label_schedule.n_desired_labels:
            new_clip = sample_segment_from_path(path, int(self._frames_per_segment))
            if new_clip:
                self.clip_manager.add(new_clip, source="on-policy callback")

        # Train our model every X episodes
        if self._episode_count % self._episodes_per_training == 0:
            self.train(iterations=self._iterations_per_training, report_frequency=25)

        # Save our model every X steps
        if self._episode_count % self._episodes_per_checkpoint == 0:
            self.save_model_checkpoint()

    def reset_training_data(self, env_id, make_env, n_pretrain_clips, clip_length, stacked_frames, workers):
        self.clip_manager.clear_old_data()

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        # We need a valid clip for the root node of our search tree.
        # Null actions are more likely to generate a valid clip than a random clip from random actions.
        first_clip = basic_segment_from_null_action(env_id, make_env, clip_length, stacked_frames)

        random_clips = segments_from_rand_rollout(
            env_id, make_env, n_desired_segments=n_pretrain_clips - 1,
            clip_length_in_seconds=clip_length, stacked_frames=stacked_frames, workers=workers)

        # Add the null-action clip first, so the root is valid.
        self.clip_manager.add(first_clip, source="null-action", sync=True)  # Make syncronus to ensure this is the first clip.
        # Now add the rest
        for clip in random_clips:
            self.clip_manager.add(clip, source="random rollout")

        self.clip_manager.sort_clips(wait_until_database_fully_sorted=True)

    def calculate_targets(self, ordinals):
        """ Project ordinal information into a cardinal value to use as a reward target """
        max_ordinal = self.clip_manager.maximum_ordinal  # Equivalent to the size of the sorting tree
        step_size = 1.0 / (max_ordinal + 1)
        offset = step_size / 2
        targets = [self._standard_deviation * stats.norm.ppf(offset + (step_size * o)) for o in ordinals]
        return targets

    def train(self, iterations=1, report_frequency=None):
        self.clip_manager.sort_clips()
        # batch_size = min(128, self.clip_manager.number_of_sorted_clips)
        _, clips, ordinals = self.clip_manager.get_sorted_clips()  # batch_size=batch_size

        obs = [clip['obs'] for clip in clips]
        acts = [clip['actions'] for clip in clips]
        targets = self.calculate_targets(ordinals)

        with self.graph.as_default():
            for i in range(1, iterations + 1):
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.obs_placeholder: np.asarray(obs),
                    self.act_placeholder: np.asarray(acts),
                    self.targets: np.asarray(targets),
                    K.learning_phase(): True
                })
                self._elapsed_training_iters += 1
                if report_frequency and i % report_frequency == 0:
                    print("%s/%s reward model training iters. (Err: %s)" % (i, iterations, loss))
                elif iterations == 1:
                    print("Reward model training iter %s (Err: %s)" % (self._elapsed_training_iters, loss))

    def _checkpoint_filename(self):
        return 'checkpoints/reward_model/%s/treesave' % (self.experiment_name)

    def save_model_checkpoint(self):
        print("Saving reward model checkpoint!")
        self.saver.save(self.sess, self._checkpoint_filename())

    def load_model_from_checkpoint(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename()))
        self.saver.restore(self.sess, filename)
        # Dump model outputs with errors
        if True:  # <-- Toggle testing with this
            with self.graph.as_default():
                clip_ids, clips, ordinals = self.clip_manager.get_sorted_clips()
                targets = self.calculate_targets(ordinals)
                for i in range(len(clips)):
                    predicted_rewards = self.sess.run(self.rewards, feed_dict={
                        self.obs_placeholder: np.asarray([clips[i]["obs"]]),
                        self.act_placeholder: np.asarray([clips[i]["actions"]]),
                        K.learning_phase(): False
                    })[0]
                    reward_sum = sum(predicted_rewards)
                    starting_reward = predicted_rewards[0]
                    ending_reward = predicted_rewards[-1]
                    print(
                        "Clip {: 3d}: predicted = {: 5.2f} | target = {: 5.2f} | error = {: 5.2f} | start = {: 5.2f} | end = {: 5.2f}"
                        .format(clip_ids[i], reward_sum, targets[i], reward_sum - targets[i], starting_reward, ending_reward))

class ComparisonRewardModel(RewardModel):
    """A model that attempts to match labeling performance on a collection of comparisons."""

    def __init__(self, env, experiment_name, summary_writer, model_type, agent_logger, label_schedule, clip_length, stacked_frames):
        if model_type == "synth":
            self.comparison_collector = SyntheticComparisonCollector()
        elif model_type == "human":
            self.comparison_collector = HumanComparisonCollector(env, experiment_name=experiment_name)
        else:
            raise ValueError("Cannot find comparison collector from keyword \"%s\"" % model_type)

        self.is_fresh = len(self.comparison_collector) == 0

        self.summary_writer = summary_writer
        self.agent_logger = agent_logger
        self.label_schedule = label_schedule
        self.experiment_name = experiment_name

        # Set up some bookkeeping
        self._frames_per_segment = clip_length * env.fps
        self._steps_since_last_training = 0
        self._steps_since_last_checkpoint = 0
        self._n_timesteps_per_model_training = 2e3  # How often should we train our model?
        self._n_timesteps_per_checkpoint = 2e4  # How often should we save our model
        self._elapsed_model_training_iters = 0
        self._num_checkpoints = 0

        # Build and initialize our model
        config = tf.ConfigProto(
            # device_count={'GPU': 0},
            # log_device_placement=True,
        )
        config.gpu_options.per_process_gpu_memory_fraction = 0.35  # allow_growth = True
        self.sess = tf.Session(config=config)

        self.obs_shape = env.observation_space.shape
        if stacked_frames > 0:
            self.obs_shape = self.obs_shape + (stacked_frames,)
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape

        self.graph = self._build_model()
        self.sess.run(tf.global_variables_initializer())
        my_vars = tf.global_variables()
        self.saver = tf.train.Saver({var.name: var for var in my_vars}, max_to_keep=0)

    def _build_model(self):
        """
        Our model takes in path segments with states and actions, and generates Q values.
        These Q values serve as predictions of the true reward.
        We can compare two segments and sum the Q values to get a prediction of a label
        of which segment is better. We then learn the weights for our model by comparing
        these labels with an authority (either a human or synthetic labeler).
        """
        # Set up observation placeholders
        self.segment_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="obs_placeholder")
        self.segment_alt_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="alt_obs_placeholder")

        # Set up action placeholders
        if self.discrete_action_space:
            self.segment_act_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, None), name="act_placeholder")
            self.segment_alt_act_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, None), name="alt_act_placeholder")
            # Discrete actions need to become one-hot vectors for the model
            segment_act = tf.one_hot(tf.cast(self.segment_act_placeholder, tf.int32), self.act_shape[0])
            segment_alt_act = tf.one_hot(tf.cast(self.segment_alt_act_placeholder, tf.int32), self.act_shape[0])
        else:
            self.segment_act_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, None) + self.act_shape, name="act_placeholder")
            self.segment_alt_act_placeholder = tf.placeholder(
                dtype=tf.float32, shape=(None, None) + self.act_shape, name="alt_act_placeholder")
            # Assume the actions are how we want them
            segment_act = self.segment_act_placeholder
            segement_alt_act = self.segment_alt_act_placeholder

        # A vanilla multi-layer perceptron maps a (state, action) pair to a reward (Q-value)
        # net = FullyConnectedMLP(self.obs_shape, self.act_shape)
        net = SimpleConvolveObservationQNet(self.obs_shape, self.act_shape)

        self.q_value = nn_predict_rewards(self.segment_obs_placeholder, segment_act, net, self.obs_shape, self.act_shape)
        alt_q_value = nn_predict_rewards(self.segment_alt_obs_placeholder, segment_alt_act, net, self.obs_shape, self.act_shape)

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        segment_reward_pred_left = tf.reduce_sum(self.q_value, axis=1)
        segment_reward_pred_right = tf.reduce_sum(alt_q_value, axis=1)
        reward_logits = tf.stack([segment_reward_pred_left, segment_reward_pred_right], axis=1)  # (batch_size, 2)

        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")

        # delta = 1e-5
        # clipped_comparison_labels = tf.clip_by_value(self.comparison_labels, delta, 1.0-delta)

        data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.labels)

        self.loss_op = tf.reduce_mean(data_loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=global_step)

        return tf.get_default_graph()

    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            q_value = self.sess.run(self.q_value, feed_dict={
                self.segment_obs_placeholder: np.asarray([path["obs"]]),
                self.segment_act_placeholder: np.asarray([path["actions"]]),
                K.learning_phase(): False
            })
        return q_value[0]

    def path_callback(self, path):
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length
        self._steps_since_last_checkpoint += path_length

        # self.agent_logger.log_episode(path)  <-- This is a huge memory problem!

        # We may be in a new part of the environment, so we take new segments to build comparisons from
        # TODO: Reduce the quantity of segments!
        # TODO: Prioritize new segements when doing comparisons!
        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        if segment and len(self.comparison_collector._segments) < 1000:
            self.comparison_collector.add_segment(segment)

        # If we need more comparisons, then we build them from our recent segments
        if len(self.comparison_collector) < self.label_schedule.n_desired_labels:
            self.comparison_collector.invent_comparison()

        # Train our model every X steps
        if self._steps_since_last_training >= self._n_timesteps_per_model_training:
            self.train()
            self._steps_since_last_training = 0

        # Save our model every X steps
        if self._steps_since_last_checkpoint >= self._n_timesteps_per_checkpoint:
            self.save_model_checkpoint()
            self._steps_since_last_checkpoint = 0

    def reset_training_data(self, env_id, make_env, n_pretrain_labels, clip_length, stacked_frames, workers):
        self.comparison_collector.clear_old_data()

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(
            env_id, make_env, n_desired_segments=n_pretrain_labels * 2,
            clip_length_in_seconds=clip_length, stacked_frames=stacked_frames, workers=workers)

        # Add segments to comparison collector
        for seg in pretrain_segments:
            self.comparison_collector.add_segment(seg)
        # Turn our random segments into comparisons
        for _ in range(n_pretrain_labels):
            self.comparison_collector.invent_comparison()
        # Label our comparisons
        self.comparison_collector.label_unlabeled_comparisons(goal=n_pretrain_labels, verbose=True)

    def _checkpoint_filename(self):
        return 'checkpoints/reward_model/%s/%08d' % (self.experiment_name, self._num_checkpoints)

    def save_model_checkpoint(self):
        print("Saving reward model checkpoint!")
        self._num_checkpoints += 1
        self.saver.save(self.sess, self._checkpoint_filename())

    def load_model_from_checkpoint(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename()))
        self.saver.restore(self.sess, filename)

    def train(self, iterations=1, report_frequency=None):
        self.comparison_collector.label_unlabeled_comparisons()

        for i in range(1, iterations + 1):
            minibatch_size = min(128, len(self.comparison_collector.labeled_decisive_comparisons))
            comparisons = random.sample(self.comparison_collector.labeled_decisive_comparisons, minibatch_size)
            left_segs = [self.comparison_collector.get_segment(comp['left']) for comp in comparisons]
            right_segs = [self.comparison_collector.get_segment(comp['right']) for comp in comparisons]

            left_obs = np.asarray([left['obs'] for left in left_segs])
            left_acts = np.asarray([left['actions'] for left in left_segs])
            right_obs = np.asarray([right['obs'] for right in right_segs])
            right_acts = np.asarray([right['actions'] for right in right_segs])
            labels = np.asarray([comp['label'] for comp in comparisons])

            with self.graph.as_default():
                _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict={
                    self.segment_obs_placeholder: left_obs,
                    self.segment_act_placeholder: left_acts,
                    self.segment_alt_obs_placeholder: right_obs,
                    self.segment_alt_act_placeholder: right_acts,
                    self.labels: labels,
                    K.learning_phase(): True
                })
                self._elapsed_model_training_iters += 1
                self._write_training_summaries(loss)

            if report_frequency and i % report_frequency == 0:
                print("%s/%s reward model training iters... " % (i, iterations))

    def _write_training_summaries(self, loss):
        self.agent_logger.log_simple("reward_model/loss", loss)

        # Calculate correlation between true and predicted reward by running validation on recent episodes
        recent_paths = self.agent_logger.get_recent_paths_with_padding()
        if len(recent_paths) > 1 and self.agent_logger.summary_step % 10 == 0:  # Run validation every 10 iters
            validation_obs = np.asarray([path["obs"] for path in recent_paths])
            validation_acts = np.asarray([path["actions"] for path in recent_paths])
            q_value = self.sess.run(self.q_value, feed_dict={
                self.segment_obs_placeholder: validation_obs,
                self.segment_act_placeholder: validation_acts,
                K.learning_phase(): False
            })
            ep_reward_pred = np.sum(q_value, axis=1)
            reward_true = np.asarray([path['original_rewards'] for path in recent_paths])
            ep_reward_true = np.sum(reward_true, axis=1)
            self.agent_logger.log_simple("reward_model/correlations", corrcoef(ep_reward_true, ep_reward_pred))

        self.agent_logger.log_simple("reward_model/num_training_iters", self._elapsed_model_training_iters)
        self.agent_logger.log_simple("labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple("labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple(
            "labels/labeled_comparisons", len(self.comparison_collector.labeled_decisive_comparisons))
