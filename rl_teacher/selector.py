from rl_teacher.segment_sampling import segments_from_rand_rollout, sample_segment_from_path, basic_segment_from_null_action
import numpy as np

class Selector(object):
    def __init__(self):
        print("Selector initialized")

    def select(self, segments):
        print("Selector.select()")
        return segments[:2], 0

class MinMaxSelector(object):
    def __init__(self):
        print("Selector initialized")

    def select(self, segments):
        print("Selector.select()")
        sort = np.argsort([sum(segment['rewards']) for segment in segments])
        return [segments[sort[0]], segments[sort[-1]]], 0

class VarianceSelector(object):
    def __init__(self):
        print("Selector initialized")

    def select(self, segments):
        print("Selector.select()")
        sort = np.argsort([segment['variance'] for segment in segments])
        return [segments[sort[-1]]], 0

class ClipSelector(object):
    """ Wraps a reward model's path_callback to sample, select and record segments for human to annotate. """

    def __init__(self, model, env_id, make_env, save_dir, paths_per_selection=500):
        self.model = model
        self.selector = MinMaxSelector()
        self.env_id = env_id
        self.make_env = make_env
        self.save_dir = save_dir

        self.paths_per_wait = 1
        self.clip_length = 90
        self.stacked_frames = 4
        self.workers = 4

        self.paths_per_selection = paths_per_selection
        self._num_paths_seen = 0  # Internal counter of how many paths we've seen

        self.collected_paths = []

    def path_callback(self, path):
        # Video recording to elicit human feedback every x steps.
        if (self._num_paths_seen % self.paths_per_wait <= self.paths_per_selection):
            if (len(self.collected_paths) < self.paths_per_selection):
                self.collected_paths.append(path)
            elif (len(self.collected_paths) == self.paths_per_selection):
                selected_paths, selection_time = self.selector.select(self.collected_paths)
                for selected_path in selected_paths:
                    segment = sample_segment_from_path(selected_path, int(self.model._frames_per_segment))
                    if segment:
                        self.model.clip_manager.add(segment, source="on-policy callback")

                self.model.clip_manager.sort_clips(wait_until_database_fully_sorted=True)
                self.collected_paths = []
                print("clips sorted.")

        self._num_paths_seen += 1
        self.model.path_callback(path)

    def predict_reward(self, path):
        return self.model.predict_reward(path)
