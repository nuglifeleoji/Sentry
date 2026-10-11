#!/usr/bin/env python3
"""Verify standard sdist/wheel contents against the supported core source tree.

This offline check does not import the runtime or read local credentials.
"""
from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_FILES = {
    "pyproject.toml", "MANIFEST.in", "README.md",
    "LICENSE", ".env.example", ".gitignore",
    "scripts/check_release.py", "configs/paper.yaml",
    "benchmarks/__init__.py", "benchmarks/appworld/__init__.py",
    "benchmarks/appworld/adapter.py", "benchmarks/appworld/run.py",
    *{f"configs/providers/{name}.yaml" for name in (
        "openai", "openai_responses", "openai_compatible", "openrouter",
        "anthropic", "gemini", "together",
    )},
}
# Recognizable credential forms are rejected without displaying matched values.
_CREDENTIAL = re.compile(
    rb"(?:sk-or-v1-[A-Za-z0-9_-]{32,}|sk-proj-[A-Za-z0-9_-]{24,}|"
    rb"sk-ant-[A-Za-z0-9_-]{24,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_public_source(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    if name in _PUBLIC_FILES:
        return True
    return (
        len(path.parts) == 2 and path.parts[0] == "Sentry" and path.suffix == ".py"
        and not path.name.startswith(".")
        and not any(word in path.stem.lower() for word in ("secret", "credential", "private"))
    )


def public_files(root: Path) -> list[Path]:
    """Read exact support/benchmark files and flat core modules, never local data."""
    root = root.resolve()
    candidates = {root / name for name in _PUBLIC_FILES}
    require(not (root / "Sentry").is_symlink(), "Runtime source directory is a symlink")
    candidates.update((root / "Sentry").glob("*.py"))
    result = []
    for path in sorted(candidates):
        if not is_public_source(path.relative_to(root).as_posix()):
            continue
        require(not any(parent.is_symlink() for parent in (path, *path.parents) if parent != root),
                f"Public source path is a symlink: {path.relative_to(root)}")
        if path.is_file():
            result.append(path)
    return result


def _safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    require(bool(path.parts) and not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            "Unsafe distribution member path")
    return path


def _wheel_files(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate wheel member")
        for member in archive.infolist():
            _safe_path(member.filename)
            require(not stat.S_ISLNK(member.external_attr >> 16), "Wheel contains a symlink")
        return {name: archive.read(name) for name in names if not name.endswith("/")}


def _sdist_files(path: Path) -> dict[str, bytes]:
    files = {}
    prefix = None
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            path = _safe_path(member.name)
            prefix = prefix or path.parts[0]
            require(path.parts[0] == prefix, "Sdist contains multiple root directories")
            if member.isdir():
                continue
            require(member.isfile() and len(path.parts) > 1, "Non-regular sdist member")
            name = PurePosixPath(*path.parts[1:]).as_posix()
            require(name not in files, "Duplicate sdist member")
            files[name] = archive.extractfile(member).read()
    return files


def _check_metadata(payload: bytes, source: dict[str, bytes], version: str) -> None:
    metadata = Parser().parsestr(payload.decode())
    require(metadata["Name"] == "sentry-failure-manager" and metadata["Version"] == version,
            "Distribution name/version differs from the project")
    require(metadata["License-Expression"] == "MIT" and metadata.get_all("License-File") == ["LICENSE"],
            "Distribution MIT license metadata is missing or changed")
    require(metadata["Requires-Python"] == ">=3.10", "Unexpected supported Python versions")
    requirements = metadata.get_all("Requires-Dist") or []
    base = [item.replace(" ", "") for item in requirements if ";" not in item]
    require(base in (["PyYAML<7,>=6"], ["PyYAML>=6,<7"]), "Unexpected base dependencies")
    appworld = [item for item in requirements if item.lower().startswith("appworld")]
    require("appworld" in (metadata.get_all("Provides-Extra") or [])
            and appworld == ['appworld==0.1.3.post1; extra == "appworld"'],
            "AppWorld must be the exact optional dependency without a Python marker")
    require(metadata.get_payload().strip() == source["README.md"].decode().strip(),
            "Distribution README differs from source")


def check_distributions(source_root: Path, wheel: Path, sdist: Path) -> dict[str, int]:
    source_root = source_root.resolve()
    source = {path.relative_to(source_root).as_posix(): path.read_bytes() for path in public_files(source_root)}
    require({"pyproject.toml", "README.md", "LICENSE", "Sentry/__init__.py"} <= source.keys(),
            "Required core project files are missing")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', source["pyproject.toml"].decode(), re.MULTILINE)
    require(version_match is not None, "Project version is missing")
    version = version_match.group(1)
    files = _sdist_files(sdist)
    public_sdist = {name: payload for name, payload in files.items() if is_public_source(name)}
    require(public_sdist == source, "Sdist public files differ from the supported source tree")
    generated = {"PKG-INFO", "setup.cfg"} | {
        f"sentry_failure_manager.egg-info/{name}" for name in (
            "PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
            "requires.txt", "top_level.txt",
        )
    }
    require(files.keys() - public_sdist.keys() <= generated, "Unexpected sdist payload")
    _check_metadata(files["PKG-INFO"], source, version)

    wheel_files = _wheel_files(wheel)
    dist_info = f"sentry_failure_manager-{version}.dist-info"
    runtime = {name: data for name, data in source.items() if name.startswith("Sentry/")}
    require({name: data for name, data in wheel_files.items() if name.startswith("Sentry/")} == runtime,
            "Wheel runtime differs from source")
    metadata_paths = {f"{dist_info}/{name}" for name in (
        "METADATA", "WHEEL", "entry_points.txt", "top_level.txt", "RECORD", "licenses/LICENSE",
    )}
    require(wheel_files.keys() - runtime.keys() <= metadata_paths, "Unexpected wheel payload")
    _check_metadata(wheel_files[f"{dist_info}/METADATA"], source, version)
    require(wheel_files.get(f"{dist_info}/licenses/LICENSE") == source["LICENSE"],
            "Wheel MIT license text is missing or changed")
    entry_points = configparser.ConfigParser()
    entry_points.read_string(wheel_files[f"{dist_info}/entry_points.txt"].decode())
    require(entry_points.sections() == ["console_scripts"]
            and dict(entry_points["console_scripts"]) == {
                "sentry-validate": "Sentry.cli:main",
            },
            "Unexpected installed commands")
    for label, payloads in (("sdist", files), ("wheel", wheel_files)):
        for name, payload in payloads.items():
            require(_CREDENTIAL.search(payload) is None, f"Credential-shaped content in {label}: {name}")
    return {"source_files": len(source), "wheel_files": len(wheel_files), "sdist_files": len(files)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    args = parser.parse_args()
    try:
        counts = check_distributions(args.source_root, args.wheel, args.sdist)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Release check failed: {exc}\n")
    print("Verified core distributions: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
