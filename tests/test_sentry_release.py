"""Distribution boundaries: only reviewed core sources and the runtime wheel."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile
from zipfile import ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sentry_release", Path(__file__).resolve().parents[1] / "scripts/check_release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
DIST_INFO = "sentry_failure_manager-0.11.0.dist-info"


def _archives(
    tmp_path, *, extra_sdist=None, extra_wheel=None, commands=None, license="MIT",
    appworld_requirement='appworld==0.1.3.post1; extra == "appworld"',
):
    source = tmp_path / "source"
    files = {
        "pyproject.toml": b'version = "0.11.0"\n',
        "README.md": b"Sentry core\n", "LICENSE": b"MIT license fixture\n",
        "Sentry/__init__.py": b"# runtime\n",
        "Sentry/cli.py": b"# validation command\n",
        "benchmarks/__init__.py": b"# source-only benchmarks\n",
        "benchmarks/appworld/__init__.py": b"# appworld benchmark\n",
        "benchmarks/appworld/adapter.py": b"# official environment adapter\n",
        "benchmarks/appworld/run.py": b"# benchmark runner\n",
    }
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    metadata = (
        "Metadata-Version: 2.4\nName: sentry-failure-manager\nVersion: 0.11.0\n"
        f"License-Expression: {license}\nLicense-File: LICENSE\nRequires-Python: >=3.10\n"
        "Requires-Dist: PyYAML<7,>=6\nProvides-Extra: appworld\n"
        f"Requires-Dist: {appworld_requirement}\n\nSentry core\n"
    ).encode()
    sdist = tmp_path / "source.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, data in {**files, "PKG-INFO": metadata, **(extra_sdist or {})}.items():
            entry = tarfile.TarInfo(f"sentry_failure_manager-0.11.0/{name}")
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    wheel = tmp_path / "runtime.whl"
    with ZipFile(wheel, "w") as archive:
        for name, data in {
            **{name: data for name, data in files.items() if name.startswith("Sentry/")},
            f"{DIST_INFO}/METADATA": metadata,
            f"{DIST_INFO}/licenses/LICENSE": files["LICENSE"],
            f"{DIST_INFO}/entry_points.txt": commands or b"[console_scripts]\nsentry-validate = Sentry.cli:main\n",
            **(extra_wheel or {}),
        }.items():
            archive.writestr(name, data)
    return source, wheel, sdist


def test_distribution_check_accepts_matching_core_artifacts(tmp_path):
    assert release.check_distributions(*_archives(tmp_path)) == {
        "source_files": 9, "wheel_files": 5, "sdist_files": 10,
    }


@pytest.mark.parametrize("name", [
    ".env", "outputs/run.json", ".legacy/experiment.py", "demo/server.py",
    "examples/agent.py", "site/index.html", "plugins/sentry/README.md",
    ".agents/plugins/marketplace.json", "benchmarks/appworld/runner.py",
    "configs/providers/secrets.yaml", "Sentry/private.py",
    "Sentry/integrations/data/tasks.json", "Sentry/integrations/local.py",
    "Sentry/integrations/appworld.py", "Sentry/integrations/__init__.py",
    "benchmarks/appworld/data/tasks.json", "benchmarks/appworld/.env",
    "configs/sentry_frozen.yaml", "configs/sentry_held_out.yaml",
    "SENTRY_ALGORITHM_CONFORMANCE.md", "docs/REPRODUCIBILITY_AUDIT.md",
    "docs/API_PROVIDERS.md", "docs/INTEGRATION.md", "docs/APPWORLD.md",
    "CONTRIBUTING.md", "SECURITY.md",
    "tests/conftest.py", "tests/test_sentry_appworld.py", "tests/test_sentry_run.py",
])
def test_sdist_rejects_files_outside_distribution_allowlist(tmp_path, name):
    with pytest.raises(ValueError, match="Unexpected sdist payload"):
        release.check_distributions(*_archives(tmp_path, extra_sdist={name: b"fixture"}))


@pytest.mark.parametrize("name", [
    "demo/__init__.py", "examples/agent.py", "configs/paper.yaml", f"{DIST_INFO}/local.json",
    "benchmarks/__init__.py", "benchmarks/appworld/__init__.py",
    "benchmarks/appworld/adapter.py", "benchmarks/appworld/run.py",
    "tests/conftest.py", "tests/test_sentry_appworld.py", "tests/test_sentry_run.py",
])
def test_wheel_rejects_files_outside_runtime_and_metadata(tmp_path, name):
    with pytest.raises(ValueError, match="Unexpected wheel payload"):
        release.check_distributions(*_archives(tmp_path, extra_wheel={name: b"fixture"}))


@pytest.mark.parametrize("name", ["Sentry/__init__.py", "Sentry/cli.py"])
def test_wheel_runtime_must_match_reviewed_source(tmp_path, name):
    with pytest.raises(ValueError, match="Wheel runtime differs"):
        release.check_distributions(*_archives(tmp_path, extra_wheel={name: b"changed runtime"}))


def test_removed_cli_cannot_return(tmp_path):
    commands = b"[console_scripts]\nsentry = Sentry.run:main\nsentry-validate = Sentry.cli:main\nsentry-demo = demo.server:main\n"
    with pytest.raises(ValueError, match="Unexpected installed commands"):
        release.check_distributions(*_archives(tmp_path, commands=commands))


def test_license_metadata_must_match_core_license(tmp_path):
    with pytest.raises(ValueError, match="MIT license metadata"):
        release.check_distributions(*_archives(tmp_path, license="MIT AND OFL-1.1"))


def test_checker_does_not_print_credential_values(tmp_path):
    token = b"sk-or-v1-" + b"a" * 64
    paths = _archives(tmp_path, extra_sdist={"sentry_failure_manager.egg-info/SOURCES.txt": token})
    with pytest.raises(ValueError, match="Credential-shaped content") as result:
        release.check_distributions(*paths)
    assert token.decode() not in str(result.value)


def test_symlink_cannot_expose_local_file_under_public_name(tmp_path):
    (tmp_path / "Sentry").mkdir()
    (tmp_path / ".env").write_text("local fixture")
    (tmp_path / "Sentry/__init__.py").symlink_to(tmp_path / ".env")
    with pytest.raises(ValueError, match="symlink"):
        release.public_files(tmp_path)


def test_sdist_symlink_is_rejected(tmp_path):
    source, wheel, sdist = _archives(tmp_path)
    with tarfile.open(sdist, "w:gz") as archive:
        entry = tarfile.TarInfo("sentry_failure_manager-0.11.0/Sentry/__init__.py")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../../.env"
        archive.addfile(entry)
    with pytest.raises(ValueError, match="Non-regular sdist member"):
        release.check_distributions(source, wheel, sdist)


def test_distribution_inventory_keeps_supported_configs_and_excludes_tests_and_docs():
    assert release.is_public_source("configs/providers/openrouter.yaml")
    assert not release.is_public_source("tests/test_sentry_memory_lifecycle.py")
    assert not release.is_public_source("docs/INTEGRATION.md")
    assert not release.is_public_source("configs/providers/local.yaml")
    assert not release.is_public_source("tests/test_sentry_demo.py")
    assert not release.is_public_source("docs/LIVE_DEMO.md")


@pytest.mark.parametrize("requirement", [
    "appworld==0.1.3.post1",
    'appworld==0.1.3.post1; python_version >= "3.11" and extra == "appworld"',
])
def test_appworld_is_optional_without_silently_skipping_unsupported_python(tmp_path, requirement):
    with pytest.raises(ValueError, match="base dependencies|exact optional dependency"):
        release.check_distributions(*_archives(tmp_path, appworld_requirement=requirement))


def test_benchmark_source_allowlist_excludes_old_integrations_and_data():
    assert release.is_public_source("benchmarks/__init__.py")
    assert release.is_public_source("benchmarks/appworld/__init__.py")
    assert release.is_public_source("benchmarks/appworld/adapter.py")
    assert release.is_public_source("benchmarks/appworld/run.py")
    assert not release.is_public_source("tests/test_sentry_appworld.py")
    assert not release.is_public_source("tests/test_sentry_run.py")
    assert not release.is_public_source("docs/APPWORLD.md")
    assert not release.is_public_source("Sentry/integrations/other.py")
    assert not release.is_public_source("Sentry/integrations/data/task.py")
    assert not release.is_public_source("Sentry/integrations/appworld.py")
    assert not release.is_public_source("benchmarks/appworld/local.py")
    assert not release.is_public_source("benchmarks/appworld/data/task.py")
