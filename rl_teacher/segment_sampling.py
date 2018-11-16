import math
from multiprocessing import Pool
import numpy as np
import gym.spaces.prng as space_prng

from rl_teacher.utils import get_timesteps_per_episode

def _slice_path(path, segment_length, start_pos=0):
    # TODO return var
    return {
        k: np.asarray(v[start_pos:(start_pos + segment_length)])
        for k, v in path.items()
        if k in ['obs', "actions", 'original_rewards', 'human_obs']}

# randomly get a clip from a trajectory
def sample_segment_from_path(path, segment_length):
    """Returns a segment sampled from a random place in a path. Returns None if the path is too short"""
    path_length = len(path["obs"])
    if path_length < segment_length:
        return None

    start_pos = np.random.randint(0, path_length - segment_length + 1)

    # Build segment
    segment = _slice_path(path, segment_length, start_pos)

    return segment

def offset_for_stacking(items, offset):
    """ Remove offset items from the end and copy out items from the start
    of the list to offset to the original length. """
    if offset < 1:
        return items
    return [items[0] for _ in range(offset)] + items[:-offset]

def stack_frames(obs, depth):
    """ Take a list of n obs arrays of shape x and stack them to return an array
    of shape (n,x[0],...,x[-1],depth). If depth=3, the first item will be just
    three copies of the first frame stacked. The second item will have two copies
    of the first frame, and one of the second. The third item will be 1,2,3.
    The fourth will be 2,3,4 and so on."""
    if depth < 1:
        # Don't stack
        return np.array(obs)
    stacked_frames = np.array([offset_for_stacking(obs, offset) for offset in range(depth)])
    # Move the stack to be at the end and return
    return np.transpose(stacked_frames, list(range(1, len(stacked_frames.shape))) + [0])

def random_action(env, ob):
    """ Pick an action by uniformly sampling the environment's action space. """
    return env.action_space.sample()

def null_action(env, ob):
    """ Do nothing. """
    if hasattr(env.action_space, 'n'):  # Is descrete
        return 0
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):  # Is box
        return (env.action_space.low + env.action_space.high) / 2.0  # Return the most average action
    raise NotImplementedError()  # TODO: Handle other action spaces

def do_rollout(env, action_function, stacked_frames):
    """ Builds a path by running through an environment using a provided function to select actions. """
    obs, rewards, actions, human_obs = [], [], [], []
    max_timesteps_per_episode = get_timesteps_per_episode(env)
    ob = env.reset()
    # Primary environment loop
    for i in range(max_timesteps_per_episode):
        action = action_function(env, ob)
        obs.append(ob)
        actions.append(action)
        ob, rew, done, info = env.step(action)
        rewards.append(rew)
        human_obs.append(info.get("human_obs"))
        if done:
            break
    # Build path dictionary
    path = {
        "obs": stack_frames(obs, stacked_frames),
        "original_rewards": np.array(rewards),
        "actions": np.array(actions),
        "human_obs": np.array(human_obs)}
    return path

def basic_segment_from_null_action(env_id, make_env, clip_length_in_seconds, stacked_frames):
    """ Returns a segment from the start of a path made from doing nothing. """
    env = make_env(env_id)
    segment_length = int(clip_length_in_seconds * env.fps)
    path = do_rollout(env, null_action, stacked_frames)
    return _slice_path(path, segment_length)

def basic_segments_from_rand_rollout(
    env_id, make_env, n_desired_segments, clip_length_in_seconds, stacked_frames,
    # These are only for use with multiprocessing
    seed=0, _verbose=True, _multiplier=1
):
    """ Generate a list of path segments by doing random rollouts. No multiprocessing. """
    segments = []
    env = make_env(env_id)
    env.seed(seed)
    space_prng.seed(seed)
    segment_length = int(clip_length_in_seconds * env.fps)
    while len(segments) < n_desired_segments:
        path = do_rollout(env, random_action, stacked_frames)
        # Calculate the number of segments to sample from the path
        # Such that the probability of sampling the same part twice is fairly low.
        segments_for_this_path = max(1, int(0.25 * len(path["obs"]) / segment_length))
        for _ in range(segments_for_this_path):
            segment = sample_segment_from_path(path, segment_length)
            if segment:
                segments.append(segment)

            if _verbose and len(segments) % 10 == 0 and len(segments) > 0:
                print("Collected %s/%s segments" % (len(segments) * _multiplier, n_desired_segments * _multiplier))

    if _verbose:
        print("Successfully collected %s segments" % (len(segments) * _multiplier))
    return segments

def segments_from_rand_rollout(env_id, make_env, n_desired_segments, clip_length_in_seconds, stacked_frames, workers):
    """ Generate a list of path segments by doing random rollouts. Can use multiple processes. """
    if workers < 2:  # Default to basic segment collection
        return basic_segments_from_rand_rollout(env_id, make_env, n_desired_segments, clip_length_in_seconds, stacked_frames)

    pool = Pool(processes=workers)
    segments_per_worker = int(math.ceil(n_desired_segments / workers))
    # One job per worker. Only the first worker is verbose.
    jobs = [
        (env_id, make_env, segments_per_worker, clip_length_in_seconds, stacked_frames, i, i == 0, workers)
        for i in range(workers)]
    results = pool.starmap(basic_segments_from_rand_rollout, jobs)
    pool.close()
    return [segment for sublist in results for segment in sublist][:n_desired_segments]
