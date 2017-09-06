import os
import multiprocessing
import pickle
import bisect
from time import sleep

import numpy as np

from rl_teacher.video import write_segment_to_video, upload_to_gcs

def _write_and_upload_video(clip, clip_id, source, render_full_obs, fps, gcs_path, video_local_path, clip_local_path):
    with open(clip_local_path, 'wb') as f:
        pickle.dump(clip, f)  # Write clip to disk
    write_segment_to_video(clip, fname=video_local_path, render_full_obs=render_full_obs, fps=fps)
    upload_to_gcs(video_local_path, gcs_path)
    return clip_id, source

def _tree_minimum(node):
    while node.left:
        node = node.left
    return node

def _tree_successor(node):
    # If we can descend, do the minimal descent
    if node.right:
        return _tree_minimum(node.right)
    # Else backtrack to either the root or the nearest point where descent is possible
    while node.parent and node == node.parent.right:
        node = node.parent
    # If we've backtracked to the root return None, else node.parent will be successor
    return node.parent


class SynthClipManager(object):
    """Like the basic ClipManager, but uses the original environment reward to sort the clips, and doesn't save/load from disk/database"""

    def __init__(self, env, experiment_name):
        self.env = env
        self.experiment_name = experiment_name
        self._sorted_clips = []  # List of lists of clips (each sublist's clips have equal reward sums)
        self._ordinal_rewards = []  # List of the reward sums for each sublist

    def clear_old_data(self):
        """Synthetic labeler doesn't use old data"""

    def add(self, new_clip, *, source="", sync=False):
        # Clips are sorted as they're added
        new_reward = sum(new_clip["original_rewards"])
        if new_reward in self._ordinal_rewards:
            index = self._ordinal_rewards.index(new_reward)
            self._sorted_clips[index].append(new_clip)
        else:
            index = bisect.bisect(self._ordinal_rewards, new_reward)
            self._ordinal_rewards.insert(index, new_reward)
            self._sorted_clips.insert(index, [new_clip])

    @property
    def total_number_of_clips(self):
        return self.number_of_sorted_clips

    @property
    def number_of_sorted_clips(self):
        return sum([len(self._sorted_clips[i]) for i in range(len(self._sorted_clips))])

    @property
    def maximum_ordinal(self):
        return len(self._sorted_clips) - 1

    def sort_clips(self, wait_until_database_fully_sorted=False):
        """Does nothing. Clips are sorted as they're added."""

    def get_sorted_clips(self, *, batch_size=None):
        clips_with_ordinals = []
        for ordinal in range(len(self._sorted_clips)):
            clips_with_ordinals += [{'clip': clip, 'ord': ordinal} for clip in self._sorted_clips[ordinal]]
        sample = np.random.choice(clips_with_ordinals, batch_size, replace=False) if batch_size else clips_with_ordinals
        return [None for _ in sample], [item['clip'] for item in sample], [item['ord'] for item in sample]


class ClipManager(object):
    """Saves/loads clips from disk, gets new ones from teacher, and syncs everything up with the database"""

    def __init__(self, env, experiment_name, workers=4):
        self.gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
        assert self.gcs_bucket, "you must specify a RL_TEACHER_GCS_BUCKET environment variable"
        assert self.gcs_bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"

        self.env = env
        self.experiment_name = experiment_name
        self._upload_workers = multiprocessing.Pool(workers)
        self._pending_upload_results = []

        self._clips = {}
        # Load clips from database and disk
        from human_feedback_api import Clip
        for clip in Clip.objects.filter(environment_id=self.env.spec.id):
            clip_id = clip.clip_tracking_id
            try:
                self._clips[clip_id] = pickle.load(open(self._pickle_path(clip_id), 'rb'))
            except FileNotFoundError:
                pass
            except Exception:
                print("Exception occurred when loading clip %s" % clip_id)
                if input("Do you want to erase this clip? (y/n)\n").startswith('y'):
                    print("Erasing clip from disk...")
                    os.remove(self._pickle_path(clip_id))
                    print("Erasing clip and all related data from database...")
                    clip.delete()
                    print("Warning: There's a chance that this simply invalidates the sort-tree. Check the human feedback api /tree/experiment_name for any experiments involving this clip.")
                    from human_feedback_api import Comparison
                    Comparison.objects.filter(media_url_1=clip.media_url).delete()
                    Comparison.objects.filter(media_url_2=clip.media_url).delete()
                    print("Invalid data deleted.\nMoving on...")
                else:
                    raise
        self._max_clip_id = max(self._clips.keys()) if self._clips else 0

        self._sorted_clips = []  # List of lists of clip_ids
        self.sort_clips()

        # Report
        if len(self._clips) < 1:
            print("Starting fresh!")
        else:
            print("Found %s old clips for this environment!" % (len(self._clips)))

    def clear_old_data(self):
        if len(self._clips) > 0:
            print("Erasing old clips FOR THE ENTIRE ENVIRONMENT of %s..." % self.env.spec.id)

        for clip_id in self._clips:
            os.remove(self._pickle_path(clip_id))

        self._sorted_clips = []
        self._clips = {}
        self._max_clip_id = 0

        from human_feedback_api import Clip
        Clip.objects.filter(environment_id=self.env.spec.id).delete()
        from human_feedback_api import SortTree
        SortTree.objects.filter(experiment_name=self.experiment_name).delete()
        from human_feedback_api import Comparison
        Comparison.objects.filter(experiment_name=self.experiment_name).delete()

    def _create_search_tree(self, seed_clip):
        from human_feedback_api import SortTree
        tree = SortTree(
            experiment_name=self.experiment_name,
            is_red=False,
        )
        tree.save()
        tree.bound_clips.add(seed_clip)

    def _assign_clip_to_search_tree(self, clip):
        from human_feedback_api import SortTree
        try:
            root = SortTree.objects.get(experiment_name=self.experiment_name, parent=None)
            root.pending_clips.add(clip)
        except SortTree.DoesNotExist:
            self._create_search_tree(clip)

    def _add_to_database(self, clip_id, source=""):
        from human_feedback_api import Clip
        clip = Clip(
            environment_id=self.env.spec.id,
            clip_tracking_id=clip_id,
            media_url=self._gcs_url(clip_id),
            source=source,
        )
        clip.save()
        self._assign_clip_to_search_tree(clip)

    def add(self, new_clip, *, source="", sync=False):
        clip_id = self._max_clip_id + 1
        self._max_clip_id = clip_id
        self._clips[clip_id] = new_clip
        # Write the clip to disk and upload
        if sync:
            uploaded_clip_id, _ = _write_and_upload_video(
                new_clip, clip_id, source, self.env.render_full_obs, self.env.fps, self._gcs_path(clip_id), self._video_path(clip_id), self._pickle_path(clip_id))
            self._add_to_database(uploaded_clip_id, source)
        else:  # async
            self._pending_upload_results.append(self._upload_workers.apply_async(_write_and_upload_video, (
                new_clip, clip_id, source, self.env.render_full_obs, self.env.fps, self._gcs_path(clip_id), self._video_path(clip_id), self._pickle_path(clip_id))))
        # Avoid memory leaks!
        self._check_pending_uploads()

    def _check_pending_uploads(self):
        # Check old pending results to see if we can clear memory and add them to the database. Also reveals errors.
        for pending_result in self._pending_upload_results:
            if pending_result.ready():
                uploaded_clip_id, uploaded_clip_source = pending_result.get(timeout=60)
                self._add_to_database(uploaded_clip_id, uploaded_clip_source)
        self._pending_upload_results = [r for r in self._pending_upload_results if not r.ready()]

    @property
    def total_number_of_clips(self):
        return len(self._clips)

    @property
    def number_of_sorted_clips(self):
        return sum([len(self._sorted_clips[i]) for i in range(len(self._sorted_clips))])

    @property
    def maximum_ordinal(self):
        return len(self._sorted_clips) - 1

    def sort_clips(self, wait_until_database_fully_sorted=False):
        from human_feedback_api import SortTree
        if wait_until_database_fully_sorted:
            print("Waiting until all clips in the database are sorted...")
            while self._pending_upload_results or SortTree.objects.filter(experiment_name=self.experiment_name).exclude(pending_clips=None):
                self._check_pending_uploads()
                sleep(10)
            print("Okay! The database seems to be sorted!")
        sorted_clips = []
        try:
            node = _tree_minimum(SortTree.objects.get(experiment_name=self.experiment_name, parent=None))
            while node:
                sorted_clips.append([x.clip_tracking_id for x in node.bound_clips.all()])
                node = _tree_successor(node)
        except SortTree.DoesNotExist:
            pass  # Root hasn't been created.
        self._sorted_clips = sorted_clips

    def get_sorted_clips(self, *, batch_size=None):
        clip_ids_with_ordinals = []
        for ordinal in range(len(self._sorted_clips)):
            clip_ids_with_ordinals += [{'id': clip_id, 'ord': ordinal} for clip_id in self._sorted_clips[ordinal]]
        sample = np.random.choice(clip_ids_with_ordinals, batch_size, replace=False) if batch_size else clip_ids_with_ordinals
        return [item['id'] for item in sample], [self._clips[item['id']] for item in sample], [item['ord'] for item in sample]

    def _video_filename(self, clip_id):
        return "%s-%s.mp4" % (self.experiment_name, clip_id)

    def _video_path(self, clip_id):
        return os.path.join('/tmp/rl_teacher_media', self._video_filename(clip_id))

    def _pickle_path(self, clip_id):
        return os.path.join('clips', '%s-%s.clip' % (self.env.spec.id, clip_id))

    def _gcs_path(self, clip_id):
        return os.path.join(self.gcs_bucket, self._video_filename(clip_id))

    def _gcs_url(self, clip_id):
        return "https://storage.googleapis.com/%s/%s" % (self.gcs_bucket.lstrip("gs://"), self._video_filename(clip_id))
