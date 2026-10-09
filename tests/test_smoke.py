import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", "venv", "__pycache__", ".git"}


def _python_files():
    return [
        p
        for p in ROOT.rglob("*.py")
        if not SKIP.intersection(p.parts) and "tests" not in p.relative_to(ROOT).parts
    ]


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_source_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize(
    "module",
    [
        "numpy",
        "pandas",
        "networkx",
        "statsmodels",
        "scipy",
        "lightgbm",
        "sklearn",
        "matplotlib",
        "shap",
    ],
)
def test_dependency_imports(module):
    importlib.import_module(module)
