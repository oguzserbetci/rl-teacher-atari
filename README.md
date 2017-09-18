# RL-Teacher-Atari

`rl-teacher-atari` is an extension of [`rl-teacher`](https://github.com/nottombrown/rl-teacher), which is in turn an implementation of of [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) [Christiano et al., 2017].

![Video](https://i.makeagif.com/media/9-14-2017/4hfT3W.gif)

As-is, `rl-teacher` only handles MuJoCo environments. This repository is meant to extend that functionality to Atari environments and other complex Gym environments. Additionally, this repository extends and augments the code in the following ways:

- Full support for Gym Atari environments
- Added [`GA3C`](https://github.com/NVlabs/GA3C) agent to optimize Atari and other complex environments
- Extended `parallel_trpo` to theoretically be able to handle environments with discrete action spaces
- Added save/load checkpoint functionality to reward models (and `GA3C`+database)
- Made `human-feedback-api` much more efficient by having humans sort clips into a red-black tree instead of doing blind comparisons
- Added a visualization of the sorting tree
- Simplified reward models by having the model minimize squared error between predicted reward and a real-number reward based on the ordering of clips in the tree
- Added support for frame-stacking
- Other miscellaneous improvements like speeding up pretraining, removing the multiprocess dependency from `parallel-trpo`, and adding the ability to define custom start-points in an Atari environment

![Red-Black Tree](https://i.imgur.com/AfFBxpy.png)

# Installation

The setup instructions are identical to [`rl-teacher`](https://github.com/nottombrown/rl-teacher#installation) except that you no longer need to set up MuJoCo unless you are trying to run MuJoCo environments, and you no longer need to install agents that are unused.

To run Atari specifically, use
```
cd ~/rl-teacher-atari
pip install -e .
pip install -e human-feedback-api
pip install -e agents/ga3c
```

# Usage

To run `rl-teacher-atari`, use the same sorts of commands that you'd use for `rl-teacher`.

Examples:
```
python rl_teacher/teach.py -e Pong-v0 -n rl-test -p rl
python rl_teacher/teach.py -e Breakout-v0 -n synth-test -p synth -l 300
python rl_teacher/teach.py -e MontezumaRevenge-v0 -n human-test -p human -L 50
```

Note that with `rl-teacher-atari` you'll need far fewer labels.
You'll also want to switch the agent back to `parallel_trpo` for solving MuJoCo environments.

```
python rl_teacher/teach.py -p rl -e ShortHopper-v1 -n base-rl -a parallel_trpo
```

![Tensorboard Graph](https://i.imgur.com/7jrAKJi.png)

There are a few new command-line arguments that are worth knowing about. Primarily, there are a set of four flags:
- `--force_new_environment_clips`
- `--force_new_training_labels`
- `--force_new_reward_model`
- `--force_new_agent_model`
Activating these flags will erase the corresponding data from the disk/database. For the most part this won't be necessary, and you can simply pick a new experiment name. Note, however, that *experiments within the same environment now share clips* so you may want to `--force_new_environment_clips` when starting a new experiment in an old environment.

Also worth noting, there's a parameter called `--stacked_frames` (`-f`) that defaults to *4*. This helps model movement that the human naturally sees in the video, but can alter how the system performs compared to `rl-teacher`. To remove frame stacking simply add `-f 0` to the command-line arguments.

## Backwards Compatibility

`rl-teacher-atari` is meant to be entirely backwards compatible, and do at least as well as `rl-teacher` on all tasks. If `rl-teacher-atari` lacks a feature that its parent has, please submit an issue.

# TODO

- [ ] Fetch in clips added under previous experiments ("two") when an old experiment ("one") is re-launched!
- [ ] Get PPO agent(s) working
- [ ] Get all agents saving/loading cleanly
- [ ] Make the reward model select the right neural net based on the shape of the environment's observation space, rather than action space
- [ ] envs.py is still pretty gnarly; needs refactoring
- [ ] The red-black tree used for sorting is set up to allow pre-sorting, where a clip is assigned to a non-root node when created. Implement this!
- [ ] Get play.py into a better state
