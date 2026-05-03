"""Tests for project packaging metadata."""

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _runtime_requirements():
    return [
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_pyproject_declares_src_layout_pytest_path_and_entrypoint():
    data = _pyproject()

    assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert data["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]
    assert data["project"]["scripts"]["cv-scribble-diffusion"] == (
        "cv_scribble_diffusion.app.startup:main"
    )


def test_pyproject_runtime_dependencies_match_requirements_file():
    data = _pyproject()

    assert data["project"]["dependencies"] == _runtime_requirements()


def test_pyproject_dev_extra_matches_dev_requirements_without_runtime_include():
    data = _pyproject()
    dev_requirements = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-r"))
    ]

    assert data["project"]["optional-dependencies"]["dev"] == dev_requirements
