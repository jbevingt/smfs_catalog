"""provenance.sklearn_version(): one shared copy, narrow exception handling."""

import re
import sys
import types
from pathlib import Path

import pytest

from smfs_catalog import provenance

_PKG = Path(__file__).resolve().parent.parent / "smfs_catalog"
_WINDOWS = ["gmm_fit_window.py", "pca_window.py"]


def test_returns_installed_sklearn_version():
    import sklearn
    assert provenance.sklearn_version() == str(sklearn.__version__)


def test_unknown_when_sklearn_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)  # import -> ImportError
    assert provenance.sklearn_version() == "unknown"


def test_unknown_when_sklearn_has_no_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))
    assert provenance.sklearn_version() == "unknown"


def test_other_exceptions_propagate(monkeypatch):
    class Broken(types.ModuleType):
        @property
        def __version__(self):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "sklearn", Broken("sklearn"))
    with pytest.raises(RuntimeError):
        provenance.sklearn_version()


@pytest.mark.parametrize("name", _WINDOWS)
def test_window_has_no_private_copy(name):
    src = (_PKG / name).read_text(encoding="utf-8")
    assert not re.search(r"\b_sklearn_version\b", src)


@pytest.mark.parametrize("name", _WINDOWS)
def test_window_uses_provenance_version(name):
    src = (_PKG / name).read_text(encoding="utf-8")
    assert re.search(r"sklearn_version\s*=\s*(?:provenance\.)?sklearn_version\(\)", src)
    assert re.search(r"\bprovenance\b", src)
