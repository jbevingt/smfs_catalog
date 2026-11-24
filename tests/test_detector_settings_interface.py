# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Pass-1 step c6: build_curve_events(sigs, band, settings, outer_events=None) and
find_outer_events(sigs, band, onset_thr, outer_threshold), with the detector
arguments grouped in a frozen DetectorSettings in roi_events.  `sigs` is
anything with .d1 .low .piezo; `band` is (lo, hi).

Mapping from the old signatures: (d1, level, piezo) -> sigs (.d1, .low,
.piezo); lo=, hi= -> band=(lo, hi); onset_thr, detector, outer_threshold,
inner_threshold, prominence, distance_pts -> DetectorSettings (defaults as
before; onset_thr has none); outer_events stays, now a plain parameter.
"""
import ast
import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smfs_catalog import roi_events
from smfs_catalog.roi_events import build_curve_events, find_outer_events

FIELD_ORDER = ["onset_thr", "detector", "outer_threshold", "inner_threshold",
               "prominence", "distance_pts"]
OLD_DEFAULTS = {"detector": "d1_threshold", "outer_threshold": 0.2,
                "inner_threshold": None, "prominence": 0.1, "distance_pts": 25}


def _sigs():
    piezo = np.arange(120, dtype=float)
    d1 = np.zeros(120, dtype=float)
    low = np.zeros(120, dtype=float)
    # One event: onset boundary at 39, loaded 40..90, inner rupture at 60,
    # terminal rupture at 90.
    low[40:91] = -1.0
    d1[60:62] = 0.35
    d1[90:93] = 0.8
    return SimpleNamespace(d1=d1, low=low, piezo=piezo)


def _settings(**kw):
    base = dict(onset_thr=-0.2, outer_threshold=0.5)
    base.update(kw)
    return roi_events.DetectorSettings(**base)


# -- interface ---------------------------------------------------------------

def test_detector_settings_has_exactly_the_six_fields_in_order():
    assert [f.name for f in dataclasses.fields(roi_events.DetectorSettings)] == FIELD_ORDER


def test_detector_settings_defaults_are_the_old_ones_and_onset_thr_is_required():
    got = {f.name: f.default for f in dataclasses.fields(roi_events.DetectorSettings)
           if f.default is not dataclasses.MISSING}
    assert got == OLD_DEFAULTS
    with pytest.raises(TypeError):
        roi_events.DetectorSettings()
    assert roi_events.DetectorSettings(-0.2).onset_thr == -0.2


def test_detector_settings_is_a_frozen_dataclass():
    ds = _settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ds.onset_thr = 1.0


def test_build_curve_events_parameters_are_exactly_the_grouped_ones():
    params = inspect.signature(build_curve_events).parameters
    assert list(params) == ["sigs", "band", "settings", "outer_events"]
    assert params["outer_events"].default is None
    assert params["settings"].default is inspect.Parameter.empty


def test_find_outer_events_parameters_are_exactly_the_grouped_ones():
    assert list(inspect.signature(find_outer_events).parameters) == [
        "sigs", "band", "onset_thr", "outer_threshold"]


# -- behaviour through the new interface ------------------------------------

def test_find_outer_events_takes_sigs_and_band():
    boundaries = find_outer_events(_sigs(), (10, 110), -0.2, 0.5)
    assert [(b.onset_idx, b.terminal_idx) for b in boundaries] == [(39, 90)]


def test_find_outer_events_band_limits_the_search():
    # The event's onset (39) lies outside a band starting at 50: no return, so no event.
    assert find_outer_events(_sigs(), (50, 110), -0.2, 0.5) == []


def test_find_outer_events_clamps_band_to_the_signal():
    boundaries = find_outer_events(_sigs(), (-5, 10_000), -0.2, 0.5)
    assert [(b.onset_idx, b.terminal_idx) for b in boundaries] == [(39, 90)]


def test_build_curve_events_inner_threshold_splits_the_event():
    events = build_curve_events(_sigs(), (10, 110), _settings(inner_threshold=0.2))
    assert events.n_rois == 1
    assert [r.idx for r in events.rois[0].ruptures] == [60, 90]
    assert events.detector == "d1_threshold"


def test_build_curve_events_inner_defaults_to_outer():
    events = build_curve_events(_sigs(), (10, 110), _settings())
    assert [r.idx for r in events.rois[0].ruptures] == [90]


def test_build_curve_events_find_peaks_detector_is_honoured():
    events = build_curve_events(
        _sigs(), (10, 110),
        _settings(detector="find_peaks", inner_threshold=0.2,
                  prominence=0.1, distance_pts=5))
    assert events.detector == "find_peaks"
    assert [r.idx for r in events.rois[0].ruptures] == [60, 90]


def test_build_curve_events_distance_pts_reaches_find_peaks():
    # Peaks at about 60 and 90 are 30 apart: a 40-point minimum distance
    # keeps only the stronger one.
    events = build_curve_events(
        _sigs(), (10, 110),
        _settings(detector="find_peaks", inner_threshold=0.2,
                  prominence=0.1, distance_pts=40))
    assert [r.idx for r in events.rois[0].ruptures] == [90]


def test_build_curve_events_uses_given_outer_events_instead_of_searching():
    sigs = _sigs()
    assert build_curve_events(sigs, (10, 110), _settings(), outer_events=[]).n_rois == 0
    found = find_outer_events(sigs, (10, 110), -0.2, 0.5)
    again = build_curve_events(sigs, (10, 110), _settings(), outer_events=found)
    searched = build_curve_events(sigs, (10, 110), _settings())
    assert roi_events.events_to_payload(again) == roi_events.events_to_payload(searched)


# -- source-size guard (CLAUDE.md limits), scoped to the two functions -------

_DECISION_NODES = (ast.If, ast.For, ast.While, ast.IfExp, ast.ExceptHandler,
                   ast.comprehension, ast.match_case)
_NEST_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With)


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
    tree = ast.parse(Path(roi_events.__file__).read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


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


@pytest.mark.parametrize("name", ["find_outer_events", "build_curve_events"])
def test_function_is_within_the_size_limits(name):
    assert _violations(name, _functions()[name]) == []
