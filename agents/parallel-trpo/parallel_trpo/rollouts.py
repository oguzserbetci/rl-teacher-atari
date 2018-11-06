import os
import multiprocessing
from time import time
from time import sleep

import numpy as np
import tensorflow as tf
from parallel_trpo.utils import filter_ob, make_network, softmax

def get_frame_stack(frames, depth):
    if depth < 1:
        # Don't stack
        return np.array(frames[-1])
    lookback_length = min(depth, len(frames) - 1)
    current_frame_copies = depth - lookback_length
    frames_list = [frames[-1] for _ in range(current_frame_copies)] + [frames[-i] for i in range(lookback_length)]
    # Reverse so the oldest frames come first instead of last
    frames_list.reverse()
    stacked_frames = np.array(frames_list)
    # Move the stack to be the last dimension and return
    return np.transpose(stacked_frames, list(range(1, len(stacked_frames.shape))) + [0])

class Actor(multiprocessing.Process):
    def __init__(self, task_q, result_q, env_id, make_env, stacked_frames, seed, max_timesteps_per_episode):
        multiprocessing.Process.__init__(self)
        self.env_id = env_id
        self.make_env = make_env
        self.stacked_frames = stacked_frames
        self.seed = seed
        self.task_q = task_q
        self.result_q = result_q

        self.max_timesteps_per_episode = max_timesteps_per_episode

    # TODO Cleanup
    def set_policy(self, weights):
        placeholders = {}
        assigns = []
        for var in self.policy_vars:
            placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            assigns.append(tf.assign(var, placeholders[var.name]))

        feed_dict = {}
        count = 0
        for var in self.policy_vars:
            feed_dict[placeholders[var.name]] = weights[count]
            count += 1
        self.session.run(assigns, feed_dict)

    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        avg_action_dist, logstd_action_dist = self.session.run(
            [self.avg_action_dist, self.logstd_action_dist], feed_dict={self.obs: obs})
        # samples the guassian distribution
        act = avg_action_dist + np.exp(logstd_action_dist) * np.random.randn(*logstd_action_dist.shape)
        act = act.ravel()
        if not self.continuous_actions:
            act = softmax(act)
        return act, avg_action_dist, logstd_action_dist

    def run(self):
        self.env = self.make_env(self.env_id)
        self.env.seed = self.seed

        self.continuous_actions = hasattr(self.env.action_space, "shape")

        # tensorflow variables (same as in model.py)
        observation_size = list(self.env.observation_space.shape)
        if self.stacked_frames > 0:
            observation_size += [self.stacked_frames]
        hidden_size = 64
        self.action_size = np.prod(self.env.action_space.shape) if self.continuous_actions else self.env.action_space.n

        # tensorflow model of the policy
        self.obs = tf.placeholder(tf.float32, [None] + observation_size)

        self.policy_vars, self.avg_action_dist, self.logstd_action_dist = make_network(
            "policy-a", self.obs, hidden_size, self.action_size)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        while True:
            next_task = self.task_q.get(block=True)
            if next_task == "do_rollout":
                # the task is an actor request to collect experience
                path = self.rollout()
                self.task_q.task_done()
                self.result_q.put(path)
            elif next_task == "kill":
                print("Received kill message for rollout process. Shutting down...")
                self.task_q.task_done()
                break
            else:
                # the task is to set parameters of the actor policy
                self.set_policy(next_task)

                # super hacky method to make sure when we fill the queue with set parameter tasks,
                # an actor doesn't finish updating before the other actors can accept their own tasks.
                sleep(0.1)
                self.task_q.task_done()

    def rollout(self):
        unstacked_obs, obs, actions, rewards = [], [], [], []
        avg_action_dists, logstd_action_dists, human_obs = [], [], []

        unstacked_obs.append(filter_ob(self.env.reset()))
        for i in range(self.max_timesteps_per_episode):
            ob = get_frame_stack(unstacked_obs, self.stacked_frames)
            action, avg_action_dist, logstd_action_dist = self.act(ob)

            obs.append(ob)
            actions.append(action)
            avg_action_dists.append(avg_action_dist)
            logstd_action_dists.append(logstd_action_dist)

            if self.continuous_actions:
                raw_ob, rew, done, info = self.env.step(action)
            else:
                choice = np.random.choice(self.action_size, p=action)
                raw_ob, rew, done, info = self.env.step(choice)
            unstacked_obs.append(filter_ob(raw_ob))

            rewards.append(rew)
            human_obs.append(info.get("human_obs"))

            if done or i == self.max_timesteps_per_episode - 1:
                path = {
                    "obs": np.array(obs),
                    "avg_action_dist": np.concatenate(avg_action_dists),
                    "logstd_action_dist": np.concatenate(logstd_action_dists),
                    "rewards": np.array(rewards),
                    "actions": np.array(actions),
                    "human_obs": np.array(human_obs)}
                return path

class ParallelRollout(object):
    def __init__(self, env_id, make_env, stacked_frames, reward_predictor, num_workers, max_timesteps_per_episode, seed):
        self.num_workers = num_workers
        self.predictor = reward_predictor

        self.tasks_q = multiprocessing.JoinableQueue()
        self.results_q = multiprocessing.Queue()

        self.actors = []
        for i in range(self.num_workers):
            new_seed = seed * 1000 + i  # Give each actor a uniquely seeded env
            self.actors.append(Actor(
                self.tasks_q, self.results_q, env_id, make_env, stacked_frames, new_seed, max_timesteps_per_episode))

        for a in self.actors:
            a.start()

        # we will start by running 20,000 / 1000 = 20 episodes for the first iteration  TODO OLD
        self.average_timesteps_in_episode = 1000

    def rollout(self, timesteps):
        start_time = time()
        # keep 20,000 timesteps per update  TODO OLD
        # TODO Run by number of rollouts rather than time
        num_rollouts = int(timesteps / self.average_timesteps_in_episode)

        for _ in range(num_rollouts):
            self.tasks_q.put("do_rollout")
        self.tasks_q.join()

        paths = []
        for _ in range(num_rollouts):
            path = self.results_q.get()

            ################################
            #  START REWARD MODIFICATIONS  #
            ################################
            path["original_rewards"] = path["rewards"]
            path["variances"], path["rewards"] = self.predictor.predict_reward(path)
            self.predictor.path_callback(path)
            ################################
            #   END REWARD MODIFICATIONS   #
            ################################

            paths.append(path)

        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)

        return paths, time() - start_time

    def set_policy_weights(self, parameters):
        for i in range(self.num_workers):
            self.tasks_q.put(parameters)
        self.tasks_q.join()

    def end(self):
        for i in range(self.num_workers):
            self.tasks_q.put("kill")
