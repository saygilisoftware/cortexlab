"""Create a clean, bounded source archive from this checkout only."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md", ".gitignore", "requirements.txt", "requirements-dev.txt", "requirements-qwen.txt", "pyproject.toml", "package.json"]
DIRECTORIES = ["backend", "frontend", "examples", "docs", "scripts", "tests", ".github"]
EXCLUDE = {"__pycache__", ".pytest_cache", ".venv", "node_modules", "recordings", "cortexlab-initial.png"}


def main():
    target = ROOT / "CortexLab-v0.1.0.zip"
    paths = [ROOT / name for name in FILES]
    for directory in DIRECTORIES:
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and not path.is_symlink() and not any(part in EXCLUDE for part in path.relative_to(ROOT).parts))
    for path in paths:
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Invalid release path: {path}")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(paths):
            archive.write(path, "cortexlab/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Release archive failed CRC verification")
        assert "cortexlab/examples/tiny-transformer.pt" in archive.namelist()
        assert "cortexlab/frontend/vendor/THREE-LICENSE.txt" in archive.namelist()
        print(f"Created {target}: {len(archive.namelist())} files, {target.stat().st_size:,} bytes; CRC verified")


if __name__ == "__main__":
    main()
