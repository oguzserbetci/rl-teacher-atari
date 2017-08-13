import numpy as np

import gym
from gym import wrappers

from ga3c.NetworkVP import NetworkVP
from ga3c.Environment import Environment
from ga3c.Config import Config as Ga3cConfig

np.set_printoptions(precision=2, linewidth=150)

if __name__ == '__main__':
    env_id = 'MontezumaRevenge-v0'
    model_names = ['left']

    Ga3cConfig.ATARI_GAME = env_id
    Ga3cConfig.MAKE_ENV_FUNCTION = gym.make
    Ga3cConfig.PLAY_MODE = True

    env = Environment()
    done = False
    command = None
    command_steps = -1

    models = {name: NetworkVP('cpu:0', name, env.get_num_actions()) for name in model_names}
    for model in models.values():
        model.load()

    while not done:
        if env.current_state is None:
            action = 0  # NO-OP while we wait for the frame buffer to fill.
        else:
            if command_steps > 0:
                command_steps -= 1
                model = models[command]
                p = model.predict_p(np.expand_dims(env.current_state, axis=0))
                action = np.argmax(p)
            else:
                if command is None:
                    print('Please input commands with a number of steps (like "left 12")')
                command, raw_steps = input().split()
                if command not in models:
                    print('Unknown command "%s"' % command)
                    continue
                command_steps = int(raw_steps)
        rew, done, info = env.step(action)
