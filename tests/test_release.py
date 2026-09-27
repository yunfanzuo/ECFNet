import hashlib
import json
from zipfile import ZipFile

from scripts.make_release import build_release


def test_release_is_reproducible_and_excludes_data(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "README.md").write_text("test repo")
    (root / "main.py").write_text("print('test')")
    for folder, filename in [("data", "recording.mat"), ("run", "checkpoint.pth"),
                             (".venv", "private.py"), ("tests/__pycache__", "test.pyc")]:
        path = root / folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"excluded")
    (root / "tests/test_example.py").write_text("assert True")
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    build_release(root, a)
    build_release(root, b)
    assert a.read_bytes() == b.read_bytes()
    with ZipFile(a) as archive:
        assert set(archive.namelist()) == {"ECFNet/README.md", "ECFNet/main.py", "ECFNet/tests/test_example.py", "ECFNet/release-manifest.json"}
        manifest = json.loads(archive.read("ECFNet/release-manifest.json"))
        for relative, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(f"ECFNet/{relative}")).hexdigest() == digest
