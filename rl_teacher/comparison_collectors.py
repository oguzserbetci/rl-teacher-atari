import multiprocessing
import traceback
import os
import uuid
import random
import pickle
from time import sleep

import numpy as np

from rl_teacher.video import write_segment_to_video, upload_to_gcs

def _write_and_upload_video(segment, env, gcs_path, video_local_path, segment_local_path):
    try:
        with open(segment_local_path, 'wb') as f:
            pickle.dump(segment, f)  # Write seg to disk
        write_segment_to_video(segment, fname=video_local_path, env=env)
        upload_to_gcs(video_local_path, gcs_path)
    except Exception:
        # Exceptions in Pool workers don't bubble up until .get() is called.
        # But _write_and_upload_video is fire-and-forget, so we need to yell if there's a problem.
        traceback.print_exc()

class SyntheticComparisonCollector(object):
    def __init__(self):
        self._comparisons = []
        self._segments = []

    def add_segment(self, seg):
        self._segments.append(seg)

    def invent_comparison(self):
        # TODO: Make this intelligent!
        comparison = {
            "left": random.randrange(len(self._segments)),
            "right": random.randrange(len(self._segments)),
            "label": None
        }
        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    def clear_old_data(self):
        pass  # Synthetic collector doesn't save/load from disk

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self, goal=None, verbose=False):
        if goal:
            while len(self) < goal:
                self.invent_comparison()
        for comp in self.unlabeled_comparisons:
            self._add_synthetic_label(comp)
        if verbose:
            print("%s synthetic labels generated... " % (len(self.labeled_comparisons)))

    def _add_synthetic_label(comparison):
        left_seg = self.get_segment(comparison['left'])
        right_seg = self.get_segment(comparison['right'])
        left_has_more_rew = np.sum(left_seg["original_rewards"]) > np.sum(right_seg["original_rewards"])

        # Mutate the comparison and give it the new label
        comparison['label'] = 0 if left_has_more_rew else 1

    def get_segment(self, index):
        return self._segments[index]

class HumanComparisonCollector(object):
    def __init__(self, env, experiment_name, workers=4):
        self.gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
        assert self.gcs_bucket, "you must specify a RL_TEACHER_GCS_BUCKET environment variable"
        assert self.gcs_bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"

        from human_feedback_api import Comparison

        self._comparisons = []
        self._segments = {}
        self._max_segment_id = 0
        self.env = env
        self.experiment_name = experiment_name
        self._upload_workers = multiprocessing.Pool(workers)

        segment_ids = set()
        # Load comparisons from database
        for comp in Comparison.objects.filter(experiment_name=experiment_name):
            left_seg_id = self._segment_id_from_url(comp.media_url_1)
            right_seg_id = self._segment_id_from_url(comp.media_url_2)
            segment_ids.add(left_seg_id)
            segment_ids.add(right_seg_id)
            self._comparisons.append({
                "left": left_seg_id,
                "right": right_seg_id,
                "id": comp.id,
                "label": None
            })
        # Load segments from disk
        self._max_segment_id = max(segment_ids)
        for seg_id in range(1, self._max_segment_id + 1):
            self._segments[seg_id] = pickle.load(open(self._pickle_path(seg_id), 'rb'))
        # Apply labels
        self.label_unlabeled_comparisons()
        # Report
        print("Found %s old comparisons (%s labeled) of %s segments (max(id)=%s) from a previous run!" % (
            len(self._comparisons), len(self.labeled_decisive_comparisons), len(self._segments), self._max_segment_id))

    def get_segment(self, index):
        return self._segments[index]

    def clear_old_data(self):
        self._comparisons = []
        self._segments = {}
        self._max_segment_id = 0

        from human_feedback_api import Comparison
        Comparison.objects.filter(experiment_name=self.experiment_name).delete()

    def _video_filename(self, segment_id):
        return "%s-%s.mp4" % (self.experiment_name, segment_id)

    def _video_path(self, segment_id):
        return os.path.join('/tmp/rl_teacher_media', self._video_filename(segment_id))

    def _pickle_path(self, segment_id):
        return os.path.join('segments', '%s-%s.segment' % (self.experiment_name, segment_id))

    def _gcs_path(self, segment_id):
        return os.path.join(self.gcs_bucket, self._video_filename(segment_id))

    def _gcs_url(self, segment_id):
        return "https://storage.googleapis.com/%s/%s" % (self.gcs_bucket.lstrip("gs://"), self._video_filename(segment_id))

    def _segment_id_from_url(self, url):
        # HACK: This is a total hack because I'm a crazy cowboy. Refactor me pls.
        return int(url[url.rfind('-') + 1:-4])  # Grab the %d from "STUFF-%d.mp4"

    def _create_comparison_in_webapp(self, left_seg_id, right_seg_id):
        """Creates a comparison DB object. Returns the db_id of the comparison"""
        from human_feedback_api import Comparison

        comparison = Comparison(
            experiment_name=self.experiment_name,
            media_url_1=self._gcs_url(left_seg_id),
            media_url_2=self._gcs_url(right_seg_id),
            response_kind='left_or_right',
            priority=1.
        )
        comparison.full_clean()
        comparison.save()
        return comparison.id

    def add_segment(self, seg):
        seg_id = self._max_segment_id + 1
        self._max_segment_id = seg_id
        self._segments[seg_id] = seg
        # Write the segment to disk and upload
        self._upload_workers.apply_async(_write_and_upload_video, (
            seg, self.env, self._gcs_path(seg_id), self._video_path(seg_id), self._pickle_path(seg_id)))

    def invent_comparison(self):
        # TODO: Make this intelligent!
        left_seg_id = random.choice(self._segments.keys())
        right_seg_id = random.choice(self._segments.keys())
        comparison_id = self._create_comparison_in_webapp(left_seg_id, right_seg_id)
        self._comparisons.append({
            "left": left_seg_id,
            "right": right_seg_id,
            "id": comparison_id,
            "label": None
        })

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self, goal=None, verbose=False):
        from human_feedback_api import Comparison

        for comparison in self.unlabeled_comparisons:
            db_comp = Comparison.objects.get(pk=comparison['id'])
            if db_comp.response == 'left':
                comparison['label'] = 0
            elif db_comp.response == 'right':
                comparison['label'] = 1
            elif db_comp.response == 'tie' or db_comp.response == 'abstain':
                comparison['label'] = 'equal'
            # If we did not match, then there is no response yet, so we just move on

        if verbose and goal:
            print("%s/%s comparisons labeled." % (len(self.labeled_comparisons), goal))
        if goal and len(self.labeled_comparisons) < int(goal * 0.75):
            if verbose:
                print("Please add labels w/ the human-feedback-api. Sleeping...")
            # Sleep for a while to give the human opportunity to label comparisons
            sleep(10)
            # Recurse until the human has labeled most of the pretraining comparisons
            self.label_unlabeled_comparisons(goal, verbose)
