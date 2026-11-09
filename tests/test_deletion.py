"""Deletion command: dry run by default, only the project's folders and its own catalog."""

from dataclasses import dataclass

import pytest

from significance import config, deletion


@dataclass
class Cat:
    catalog_id: str
    catalog_name: str
    owned: bool
    status: str = "ready"


class FakeClient:
    def __init__(self, catalogs):
        self.catalogs = list(catalogs)
        self.deleted = []

    def list_catalogs(self):
        return list(self.catalogs)

    def delete_catalog(self, catalog_id):
        self.deleted.append(catalog_id)


CATALOGS = [
    Cat("cat_mine", config.CATALOG_NAME, owned=True),
    Cat("cat_shared_same_name", config.CATALOG_NAME, owned=False),
    Cat("cat_other", "something_else", owned=True),
    Cat("retail_catalog_500k", "retail_catalog_500k", owned=False),
]


def _project(tmp_path):
    root = tmp_path / "proj"
    for rel in (
        ".temp/data/raw",
        ".temp/work/records",
        ".temp/cache",
        "dataset",
        "outputs",
        ".temp/env_setup",
        "src",
    ):
        (root / rel).mkdir(parents=True)
    (root / ".temp/data/raw/x.zip").write_text("data")
    (root / "outputs/labels.parquet").write_text("labels")
    (root / ".temp/env_setup/devenv.sh").write_text("keep")
    (root / "src/code.py").write_text("keep")
    return root


def test_dry_run_lists_and_deletes_nothing(tmp_path):
    root = _project(tmp_path)
    client = FakeClient(CATALOGS)
    report = deletion.run(root, client)
    assert report.local == [".temp/data", ".temp/work", ".temp/cache", "dataset", "outputs"]
    assert report.catalogs == [f"{config.CATALOG_NAME} (cat_mine)"]
    assert client.deleted == []
    assert (root / ".temp/data/raw/x.zip").exists()
    assert any("would delete" in line for line in report.lines())


def test_confirm_deletes_only_project_folders_and_own_catalog(tmp_path):
    root = _project(tmp_path)
    client = FakeClient(CATALOGS)
    report = deletion.run(root, client, confirm=True)
    assert client.deleted == ["cat_mine"]
    for rel in deletion.LOCAL_TARGETS:
        assert not (root / rel).exists()
    assert (root / ".temp/env_setup/devenv.sh").exists()
    assert (root / "src/code.py").exists()
    assert report.deleted


def test_catalog_name_may_carry_the_file_suffix():
    assert deletion.is_project_catalog(Cat("c", f"{config.CATALOG_NAME}.parquet", True))
    assert not deletion.is_project_catalog(Cat("c", f"{config.CATALOG_NAME}.parquet", False))
    assert not deletion.is_project_catalog(Cat("c", f"{config.CATALOG_NAME}_v2", True))


def test_without_client_local_folders_are_still_handled(tmp_path):
    root = _project(tmp_path)
    report = deletion.run(root, None, confirm=True)
    assert report.catalogs == [] and report.skipped
    assert not (root / "outputs").exists()


def test_symlinked_target_is_removed_as_a_link_not_followed(tmp_path):
    root = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep")
    import shutil

    shutil.rmtree(root / ".temp/cache")
    (root / ".temp/cache").symlink_to(outside, target_is_directory=True)
    deletion.run(root, None, confirm=True)
    assert not (root / ".temp/cache").exists()
    assert (outside / "precious.txt").read_text() == "keep"


def test_nothing_to_delete(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    report = deletion.run(root, FakeClient([]))
    assert report.lines()[-1] == "nothing to delete"
    assert any(config.HF_DATASET_REPO in line for line in report.lines())


def test_target_folder_resolving_outside_project_is_refused(tmp_path):
    root = tmp_path / "proj2"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    (outside / "data").mkdir(parents=True)
    (root / ".temp").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="outside the project"):
        deletion.run(root, None, confirm=True)
    assert (outside / "data").exists()


def test_every_report_reminds_of_the_private_hugging_face_copy(tmp_path):
    root = _project(tmp_path)
    for confirm in (False, True):
        report = deletion.run(root, FakeClient(CATALOGS), confirm=confirm)
        assert any(config.HF_DATASET_REPO in s and "by hand" in s for s in report.skipped)
