import os
import multiprocessing
import pickle
from time import sleep

from rl_teacher.video import write_segment_to_video, upload_to_gcs

def _write_and_upload_video(clip, clip_id, source, render_full_obs, fps, gcs_path, video_local_path, clip_local_path):
    with open(clip_local_path, 'wb') as f:
        pickle.dump(clip, f)  # Write clip to disk
    write_segment_to_video(clip, fname=video_local_path, render_full_obs=render_full_obs, fps=fps)
    upload_to_gcs(video_local_path, gcs_path)
    return clip_id, source

class ClipManager(object):
    """Saves/loads clips from disk, gets new ones from teacher, and syncs everything up with the database"""

    def __init__(self, env, experiment_name, workers=4):
        self.gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
        assert self.gcs_bucket, "you must specify a RL_TEACHER_GCS_BUCKET environment variable"
        assert self.gcs_bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"

        self._clips = {}
        self._max_clip_id = 0

        self.env = env
        self.experiment_name = experiment_name
        self._upload_workers = multiprocessing.Pool(workers)
        self._pending_upload_results = []

        # TODO: Load clips from disk/database!
        raise NotImplementedError()

    def clear_old_data(self):
        if len(self._clips) > 0:
            print("Erasing old clips FOR THE ENTIRE ENVIRONMENT of %s..." % self.env.spec.id)

        for clip_id in self._clips:
            os.remove(self._pickle_path(clip_id))

        self._clips = {}
        self._max_clip_id = 0

        from human_feedback_api import Clip
        Clip.objects.filter(environment_id=self.env.spec.id).delete()

    def _add_to_database(self, clip_id, source=""):
        from human_feedback_api import Clip

        clip = Clip(
            environment_id=self.env.spec.id,
            clip_tracking_id=clip_id,
            media_url=self._gcs_url(clip_id),
            source=source,
        )
        clip.save()

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
        # Avoid memory leaks! Check old pending results to see if we can clear the memory. Also reveals errors.
        for pending_result in self._pending_upload_results:
            if pending_result.ready():
                uploaded_clip_id, uploaded_clip_source = pending_result.get(timeout=60)
                self._add_to_database(uploaded_clip_id, uploaded_clip_source)

    def sort_clips(self, wait_until_database_sorted=False):
        if wait_until_database_sorted:
            print("Waiting until all clips in the database are sorted...")
            while True:
                sleep(10)

    def _video_filename(self, clip_id):
        return "%s-%s.mp4" % (self.experiment_name, clip_id)

    def _video_path(self, clip_id):
        return os.path.join('/tmp/rl_teacher_media', self._video_filename(clip_id))

    def _pickle_path(self, clip_id):
        return os.path.join('clips', '%s-%s.clip' % (self.experiment_name, clip_id))

    def _gcs_path(self, clip_id):
        return os.path.join(self.gcs_bucket, self._video_filename(clip_id))

    def _gcs_url(self, clip_id):
        return "https://storage.googleapis.com/%s/%s" % (self.gcs_bucket.lstrip("gs://"), self._video_filename(clip_id))
