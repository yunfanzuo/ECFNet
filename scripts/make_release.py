"""Create a deterministic source-only GitHub upload archive (standard library)."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT_FILES = {
    "README.md", "CITATION.cff", "pyproject.toml", "requirements.txt", "uv.lock",
    ".gitignore", ".gitattributes", ".editorconfig", ".python-version", "main.py", "cross_validation.py",
    "datapipe.py", "dataset.py", "train.py", "visualize.py",
}
SOURCE_DIRS = {"config", "docs", "net", "utils", "tests", "scripts", ".github"}
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".txt", ".cff"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv", "data", "run", "dist"}


def collect_files(root):
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for dirname in SOURCE_DIRS:
        directory = root / dirname
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if (path.is_file() and not path.is_symlink()
                    and not EXCLUDED_PARTS.intersection(relative.parts)
                    and path.suffix in SUFFIXES):
                paths.append(path)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def write_entry(archive, name, content):
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def build_release(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    paths = collect_files(root)
    if not paths or not (root / "README.md").is_file():
        raise ValueError("No companion repository found")
    if output in paths:
        raise ValueError("Output archive must not overwrite a source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"format": 1, "root": "ECFNet", "files": {}}
    with ZipFile(output, "w") as archive:
        for path in paths:
            content = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            write_entry(archive, f"ECFNet/{relative}", content)
            manifest["files"][relative] = hashlib.sha256(content).hexdigest()
        write_entry(archive, "ECFNet/release-manifest.json",
                    (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return len(paths), digest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "dist/ECFNet-github.zip")
    args = parser.parse_args()
    count, digest = build_release(root, args.output)
    print(f"Created {args.output}: {count} source files; SHA-256 {digest}")


if __name__ == "__main__":
    main()
