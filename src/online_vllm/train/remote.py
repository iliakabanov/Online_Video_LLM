"""Copies of feature shards on Google Drive through rclone (https://rclone.org).

The remote (`rclone config`, type drive) and the Drive folder id come from the training config's features section;
a split's directory <DATA_ROOT>/features/<name>/<split> maps to <folder>/<name>/<split> on Drive.
"""
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
UPLOADED = "uploaded.txt"  # names of the shards whose Drive copy is complete


class DriveUploader:
    """Copies closed shards in the background, one at a time, and deletes each local copy once its upload succeeded.

    A failed upload raises on the next `submit` or `wait`, and the shard stays on disk; uploading it again is safe
    (rclone copies only what is missing)."""

    def __init__(self, remote: str, folder: str, local_split: Path, keep_local: bool = False):
        if shutil.which("rclone") is None:
            raise RuntimeError("rclone is not installed: https://rclone.org/install/")
        self.remote, self.folder, self.split, self.keep_local = remote, folder, Path(local_split), keep_local
        self.prefix = f"{self.split.parent.name}/{self.split.name}"
        self.job = None

    def _command(self, verb, source, target):
        return ["rclone", verb, str(source), f"{self.remote}:{self.prefix}{target}", "--drive-root-folder-id",
                self.folder]

    def uploaded(self) -> set[str]:
        path = self.split / UPLOADED
        return set(path.read_text().split()) if path.exists() else set()

    def put(self, path: Path):
        """Copy one small file (meta.json, done) right away."""
        subprocess.run(self._command("copyto", path, f"/{path.name}"), check=True, capture_output=True)

    def submit(self, shard: Path):
        self.wait()
        log = open(self.split / "upload.log", "a")
        self.job = (shard, log, subprocess.Popen(self._command("copy", shard, f"/{shard.name}"), stdout=log,
                                                 stderr=subprocess.STDOUT))
        logger.info("Uploading %s", shard)

    def wait(self):
        if self.job is None:
            return
        shard, log, process = self.job
        code = process.wait()
        log.close()
        self.job = None
        if code:
            raise RuntimeError(f"Upload of {shard} failed (rclone exit code {code}, see {self.split / 'upload.log'}); "
                               "the local copy is kept, rerun to retry")
        with open(self.split / UPLOADED, "a") as f:
            f.write(shard.name + "\n")
        if not self.keep_local:
            shutil.rmtree(shard)
        logger.info("Uploaded %s%s", shard, "" if self.keep_local else " and removed the local copy")


def download(remote: str, folder: str, local_split: Path):
    """Fetch a split's shards and metadata from Drive into local_split (only what is missing)."""
    local_split = Path(local_split)
    prefix = f"{local_split.parent.name}/{local_split.name}"
    subprocess.run(["rclone", "copy", f"{remote}:{prefix}", str(local_split), "--drive-root-folder-id", folder,
                    "--progress"], check=True)
