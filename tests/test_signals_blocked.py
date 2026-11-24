"""signals_blocked(*widgets): block signals, then restore each widget's own prior state.

The behaviour tests use a stub that records blockSignals calls (no Qt needed).
The source guard checks that no hand-written blockSignals(True)/(False) call is
left in smfs_catalog/*.py outside the helper itself.
"""
import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "smfs_catalog"


def _signals_blocked():
    from smfs_catalog.qt_utils import signals_blocked
    return signals_blocked


class StubWidget:
    """Mimics QObject.blockSignals: sets the flag, returns the previous one."""

    def __init__(self, blocked=False):
        self.blocked = blocked
        self.calls = []

    def blockSignals(self, flag):
        self.calls.append(flag)
        prev = self.blocked
        self.blocked = flag
        return prev


def test_blocked_inside_and_restored_after_normal_exit():
    w = StubWidget()
    with _signals_blocked()(w):
        assert w.blocked is True
    assert w.blocked is False


def test_restored_after_exception_and_exception_propagates():
    w = StubWidget()
    with pytest.raises(ValueError):
        with _signals_blocked()(w):
            assert w.blocked is True
            raise ValueError("boom")
    assert w.blocked is False


def test_already_blocked_widget_stays_blocked_after_inner_with():
    w = StubWidget(blocked=True)
    with _signals_blocked()(w):
        assert w.blocked is True
    assert w.blocked is True


def test_nested_on_same_widget_inner_exit_keeps_outer_block():
    w = StubWidget()
    sb = _signals_blocked()
    with sb(w):
        with sb(w):
            pass
        assert w.blocked is True
    assert w.blocked is False


def test_multiple_widgets_all_blocked_inside_each_restored_to_own_state():
    a, b = StubWidget(blocked=False), StubWidget(blocked=True)
    with _signals_blocked()(a, b):
        assert a.blocked is True and b.blocked is True
    assert a.blocked is False
    assert b.blocked is True


def test_multiple_widgets_restored_after_exception():
    a, b = StubWidget(blocked=False), StubWidget(blocked=True)
    with pytest.raises(RuntimeError):
        with _signals_blocked()(a, b):
            raise RuntimeError("boom")
    assert a.blocked is False
    assert b.blocked is True


def _hand_written_calls():
    """(file, line) of every .blockSignals(True|False) call outside signals_blocked."""
    found = []
    for path in sorted(PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        exempt = set()
        if path.name == "qt_utils.py":
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "signals_blocked":
                    exempt.update(id(n) for n in ast.walk(node))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "blockSignals"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, bool)
                    and id(node) not in exempt):
                found.append((path.name, node.lineno))
    return found


def test_no_hand_written_blocksignals_calls_remain():
    found = _hand_written_calls()
    files = sorted({f for f, _ in found})
    assert not found, (
        f"{len(found)} hand-written blockSignals(True/False) calls in "
        f"{len(files)} files; use qt_utils.signals_blocked: "
        + ", ".join(f"{f}:{n}" for f, n in found)
    )
