"""Private Hugging Face copy: privacy check, upload scope, verification, card (fake Hub API)."""

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from significance import publish

META = {
    "splits": {
        "train": {"records": 3, "rows": 9, "attended_rows": 4},
        "validation": {"records": 2, "rows": 6, "attended_rows": 2},
    },
    "columns": {"record_id": {"type": "int64", "meaning": "created: the episode"}},
    "record_id_ranges": [
        {
            "record_ids": "0-1",
            "records": 2,
            "file": "validation",
            "origin": "o",
            "in_evaluation": True,
        },
        {"record_ids": "2-4", "records": 3, "file": "train", "origin": "s", "in_evaluation": False},
    ],
    "method": {"model": "m", "history": "h", "variant": "V1o", "decoding": "d", "k_rule": "k"},
}


@dataclass
class RepoFile:
    path: str
    blob_id: str
    lfs: object = None


class FakeApi:
    def __init__(self, private=True, exists=True):
        self.private, self.exists = private, exists
        self.uploads, self.remote = [], {}

    def repo_info(self, repo_id, repo_type):
        assert repo_type == "dataset"
        if not self.exists:
            raise FileNotFoundError(repo_id)
        return SimpleNamespace(private=self.private)

    def upload_folder(self, repo_id, folder_path, repo_type, allow_patterns, commit_message):
        self.uploads.append(
            dict(
                repo_id=repo_id,
                folder=folder_path,
                repo_type=repo_type,
                allow=allow_patterns,
                message=commit_message,
            )
        )

    def get_paths_info(self, repo_id, paths, repo_type):
        return [self.remote[p] for p in paths if p in self.remote]


def _folder(tmp_path):
    for name in publish.UPLOAD_FILES:
        (tmp_path / name).write_bytes(f"content of {name}".encode())
    (tmp_path / "secret.env").write_text("never")
    return tmp_path


def test_check_private_refuses_public_and_missing_repos():
    assert publish.check_private(FakeApi(), "u/r").private
    with pytest.raises(publish.PublishError, match="not private"):
        publish.check_private(FakeApi(private=False), "u/r")
    with pytest.raises(publish.PublishError, match="cannot read"):
        publish.check_private(FakeApi(exists=False), "u/r")


def test_upload_sends_exactly_the_dataset_files(tmp_path):
    api = FakeApi()
    publish.upload(api, _folder(tmp_path), "u/r", "msg")
    (call,) = api.uploads
    assert call["allow"] == list(publish.UPLOAD_FILES) and call["repo_type"] == "dataset"
    assert "secret.env" not in call["allow"]
    (tmp_path / "train.parquet").unlink()
    with pytest.raises(publish.PublishError, match="missing"):
        publish.upload(api, tmp_path, "u/r", "msg")


def test_verify_compares_lfs_sha256_and_git_blob_ids(tmp_path):
    folder = _folder(tmp_path)
    api = FakeApi()
    sha = hashlib.sha256((folder / "train.parquet").read_bytes()).hexdigest()
    api.remote = {
        "train.parquet": RepoFile("train.parquet", "x", SimpleNamespace(sha256=sha)),
        "validation.parquet": RepoFile("validation.parquet", "x", SimpleNamespace(sha256="bad")),
        "README.md": RepoFile("README.md", publish.git_blob_sha1(folder / "README.md")),
    }
    assert publish.verify(api, folder, "u/r") == {
        "README.md": True,
        "train.parquet": True,
        "validation.parquet": False,
        "metadata.json": False,  # not uploaded
    }


def test_git_blob_sha1_matches_git():
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(b"hello\n")
    try:
        expected = subprocess.run(
            ["git", "hash-object", f.name], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert publish.git_blob_sha1(Path(f.name)) == expected
    finally:
        Path(f.name).unlink()


def test_card_has_splits_licence_and_privacy_notice():
    card = publish.render_card(META)
    assert card.startswith("---\n") and "license: other" in card
    assert "split: train" in card and "path: validation.parquet" in card
    assert "no distribution" in card and "Do not share" in card
    assert "| 2-4 | 3 | train | s | no |" in card
