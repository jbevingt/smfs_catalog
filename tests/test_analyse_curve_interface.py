# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 step c4: analyse_curve takes (curve, p, *, cache=None) with a CacheContext
instead of four loose cache keywords, and is split into named stages.  Size
limits apply to analyse_curve and every top-level helper c4 introduces in
curve_analysis.py.
"""
import ast
import inspect
from pathlib import Path

import pytest

from smfs_catalog import curve_analysis
from smfs_catalog.curve_analysis import analyse_curve

OLD_CACHE_KEYWORDS = ("db_path", "code_ver", "file_id", "conn", "param_set")


def test_analyse_curve_takes_curve_p_and_a_keyword_only_cache():
    params = inspect.signature(analyse_curve).parameters
    assert list(params) == ["curve", "p", "cache"]
    assert params["cache"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["cache"].default is None
    assert len(params) <= 5


@pytest.mark.parametrize("old", OLD_CACHE_KEYWORDS)
def test_old_cache_keywords_are_gone(old):
    assert old not in inspect.signature(analyse_curve).parameters


def test_old_cache_keywords_are_rejected_at_call_time():
    with pytest.raises(TypeError):
        analyse_curve(object(), object(), db_path="x.db", code_ver="v", file_id=1)


# ── source-size guard (CLAUDE.md limits), scoped to what c4 touches ──────────

_DECISION_NODES = (ast.If, ast.For, ast.While, ast.IfExp, ast.ExceptHandler,
                   ast.comprehension, ast.match_case)
_NEST_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With)
# Every top-level function curve_analysis.py had before c4.  Any other
# top-level function is a helper c4 introduced and is held to the same limits.
PRE_C4_FUNCTIONS = {
    "pipeline_params_from", "analyse_curve", "current_signature",
    "_lane_no_pipeline_yet", "_lane_force_extension", "analyse_file",
    "_persist_multi_event_roi",
}


def _decision_points(fn) -> int:
    n = 0
    for node in ast.walk(fn):
        if isinstance(node, _DECISION_NODES):
            n += 1
        elif isinstance(node, ast.BoolOp):
            n += len(node.values) - 1
    return n


def _depth(node, d=0) -> int:
    deepest = d
    for child in ast.iter_child_nodes(node):
        deepest = max(deepest, _depth(child, d + 1 if isinstance(child, _NEST_NODES) else d))
    return deepest


def _arg_count(fn) -> int:
    a = fn.args
    return len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) + bool(a.vararg) + bool(a.kwarg)


def _functions() -> dict:
    tree = ast.parse(Path(curve_analysis.__file__).read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _new_helpers() -> dict:
    return {n: f for n, f in _functions().items() if n not in PRE_C4_FUNCTIONS}


def _violations(name, fn) -> list:
    found = []
    if fn.end_lineno - fn.lineno + 1 > 50:
        found.append(f"{name}: more than 50 lines")
    if _decision_points(fn) > 10:
        found.append(f"{name}: more than 10 decision points")
    if _depth(fn) > 4:
        found.append(f"{name}: nesting deeper than 4")
    if _arg_count(fn) > 5:
        found.append(f"{name}: more than 5 arguments")
    return found


def test_analyse_curve_is_split_into_named_stages():
    assert len(_new_helpers()) >= 4


def test_analyse_curve_is_within_the_size_limits():
    assert _violations("analyse_curve", _functions()["analyse_curve"]) == []


def test_every_new_helper_is_within_the_size_limits():
    found = [v for n, f in _new_helpers().items() for v in _violations(n, f)]
    assert found == []
