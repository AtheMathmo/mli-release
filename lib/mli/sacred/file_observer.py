import errno
import os

from sacred.observers import FileStorageObserver


class SlurmFileStorageObserver(FileStorageObserver):
    def _make_dir(self, _id):
        new_dir = os.path.join(self.basedir, str(_id))
        os.mkdir(new_dir)
        # set only if mkdir is successful
        self.dir = new_dir

    def _make_run_dir(self, _id):
        slurm_id = os.getenv("SLURM_JOB_ID")
        if slurm_id is None:
            slurm_id = os.getenv("SLURM_ARRAY_JOB_ID")

        os.makedirs(self.basedir, exist_ok=True)
        self.dir = None

        # We're not using slurm
        if slurm_id is None:
            if _id is None:
                fail_count = 0
                _id = self._maximum_existing_run_id() + 1
                while self.dir is None:
                    try:
                        self._make_dir(_id)
                    except FileExistsError:  # Catch race conditions
                        if fail_count < 1000:
                            fail_count += 1
                            _id += 1
                        else:  # expect that something else went wrong
                            raise
            else:
                self.dir = os.path.join(self.basedir, str(_id))
                os.mkdir(self.dir)
        else:
            self.dir = os.path.join(self.basedir, slurm_id)
            try:
                os.makedirs(self.dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
