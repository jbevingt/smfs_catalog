# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The transform path and the own fit's changed role (handover checks 4 and 6).

Every force and extension divides by ONE number, the divisor.  Own fit: the
file's stored slope, untouched.  Ensemble: -(instrument nm/V) / (group value).
This file runs the real pipeline on a synthetic curve (tests/invols_synth.py:
noiseless contact of slope -0.95, one rupture) and checks:

  * compute_curve_events_coords divides by the divisor and reports it as
    CurveEventsResult.invols; under an ensemble it never needs the own fit;
  * analyse_curve no longer treats a failed own fit as fatal under an
    ensemble, and keeps each file's own Transform InvOLS row up to date
    without ever replacing an applied ensemble row;
  * base_2dh_window.py is untouched by this feature.

Expected values are hand calculations: a divisor is -199.14 / T, a force is
proportional to 1 / |divisor|, so T = 210 against a slope of -0.95 scales
every force by 0.95 / 0.9482857 = 1.00181, and the own value is
199.14 / 0.95 = 209.62105.  Every call passes an explicit temporary catalog
and the code version is pinned, so no git checkout is needed for the numbers.
"""
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import invols_synth as synth                                  # noqa: E402
from smfs_catalog import curve_analysis, db, roi_events, roi_pipeline  # noqa: E402

CODE = synth.CODE
INSTRUMENT = 199.14
OWN_SLOPE = -0.95
OWN_VALUE = INSTRUMENT / 0.95          # 209.62105..., hand calculation
GROUP_T = 210.0
DIVISOR_T = -INSTRUMENT / GROUP_T      # -0.9482857...
SCALE_T = 0.95 / 0.9482857             # 1.00181
WINDOW_TOO_LONG = 5000                 # the approach has 3000 samples
TRANSFORM = "transform_invols_nm_v"
ROOT = Path(__file__).resolve().parents[1]


# ── helpers and fixtures ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def pinned_version(monkeypatch):
    """The code version every module stamps and checks, fixed so these tests
    need no git checkout (invols_synth writes its rows under the same text)."""
    for module in (curve_analysis, roi_pipeline):
        monkeypatch.setattr(module, "cache_version", lambda: CODE)


def _setup(tmp_path, instrument=INSTRUMENT, group=None):
    """A one-file catalog; `group` (nm/V) assigns the file to a group of that value."""
    db_path, ids = synth.make_catalog(
        tmp_path, [{"instrument": instrument, "curve_type": "continuous_stretch"}])
    if group is not None:
        synth.add_group(db_path, ids, group)
    return SimpleNamespace(db=db_path, fid=ids[0])


def _set_window(db_path, points):
    db.update_analysis_param("invols_window_pts", points, db_path)


def _analyse(curve, c):
    p = curve_analysis.pipeline_params_from(db.load_analysis_params(c.db))
    return curve_analysis.analyse_curve(
        curve, p, db_path=c.db, code_ver=CODE, file_id=c.fid)


def _event_params(db_path):
    return roi_pipeline.event_params_from(db.load_analysis_params(db_path))


def _events(curve, c, stage1=None):
    """compute_curve_events_coords the way the pipeline calls it."""
    return roi_pipeline.compute_curve_events_coords(
        curve, _event_params(c.db), db_path=c.db, code_ver=CODE,
        file_id=c.fid, stage1=stage1,
        param_set=db.load_analysis_params(c.db))


def _historical(curve, c):
    """The historical call shape: no catalog, everything fresh."""
    return roi_pipeline.compute_curve_events_coords(curve, _event_params(c.db))


def _ruptures(result):
    return [r for roi in result.events.rois for r in roi.ruptures]


def _forces(result):
    return np.array([r.force_pN for r in _ruptures(result)], dtype=float)


def _extensions(result):
    return np.array([r.extension_nm for r in _ruptures(result)], dtype=float)


def _payload_text(result):
    return json.dumps(roi_events.events_to_payload(result.events), sort_keys=True)


def _stored_slope(c):
    rows = synth.fetch(c.db, "SELECT value FROM analysis_results WHERE "
                       "file_id = ? AND analysis_type = 'invols_slope'", (c.fid,))
    assert len(rows) == 1
    return rows[0][0]


def _slope_rows(c):
    return synth.fetch(c.db, "SELECT value FROM analysis_results WHERE "
                       "file_id = ? AND analysis_type = 'invols_slope'", (c.fid,))


def _transform_rows(c):
    """Every stored Transform InvOLS row of the file, whole."""
    return synth.fetch(
        c.db, "SELECT value, params_json, code_version, computed_at FROM "
        "analysis_results WHERE file_id = ? AND analysis_type = ?",
        (c.fid, TRANSFORM))


def _transform_row(c):
    rows = _transform_rows(c)
    assert len(rows) == 1, rows
    value, params_json, code_version, _computed_at = rows[0]
    return value, params_json, code_version


def _seed_applied_row(c, group_id):
    """The row an apply leaves: the group's value under an older code version,
    so any rewrite of it shows."""
    synth.put_result(c.db, c.fid, TRANSFORM, GROUP_T,
                     json.dumps({"invols_group": group_id}), "APPLIED")


# ── compute_curve_events_coords: the divisor applied ─────────────────────────

def test_own_fit_divisor_equals_the_stored_slope_exactly(tmp_path):
    c = _setup(tmp_path)
    curve = synth.make_force_curve()
    _analyse(curve, c)
    result = _events(curve, c)
    assert result.invols == _stored_slope(c)


@pytest.mark.parametrize("use_stage1", [False, True], ids=["cold", "same-pass"])
def test_own_fit_numbers_do_not_change_by_one_bit(tmp_path, use_stage1):
    c = _setup(tmp_path)
    curve = synth.make_force_curve()
    _, stage1 = _analyse(curve, c)
    reference = _historical(curve, c)
    assert len(_ruptures(reference)) == 1
    result = _events(curve, c, stage1 if use_stage1 else None)
    assert result.invols == reference.invols == _stored_slope(c)
    assert np.array_equal(_forces(result), _forces(reference))
    assert np.array_equal(_extensions(result), _extensions(reference))
    assert _payload_text(result) == _payload_text(reference)


@pytest.mark.parametrize("use_stage1", [False, True], ids=["cold", "same-pass"])
def test_an_ensemble_equal_to_the_own_value_gives_the_own_forces(
        tmp_path, use_stage1):
    c = _setup(tmp_path, group=OWN_VALUE)
    curve = synth.make_force_curve()
    _, stage1 = _analyse(curve, c)
    reference = _historical(curve, c)
    result = _events(curve, c, stage1 if use_stage1 else None)
    assert len(_ruptures(result)) == 1
    assert result.invols == pytest.approx(OWN_SLOPE, rel=1e-12)
    assert _forces(result) == pytest.approx(_forces(reference), rel=1e-12)
    assert _extensions(result) == pytest.approx(_extensions(reference), rel=1e-12)


@pytest.mark.parametrize("use_stage1", [False, True], ids=["cold", "same-pass"])
def test_an_ensemble_value_of_210_scales_every_force_by_1_00181(
        tmp_path, use_stage1):
    c = _setup(tmp_path, group=GROUP_T)
    curve = synth.make_force_curve()
    _, stage1 = _analyse(curve, c)
    reference = _historical(curve, c)
    result = _events(curve, c, stage1 if use_stage1 else None)
    assert result.invols == pytest.approx(DIVISOR_T, rel=1e-12)
    assert _forces(result) / _forces(reference) == pytest.approx(
        SCALE_T, rel=1e-5)


def test_an_ensemble_file_does_not_need_its_own_fit(tmp_path):
    c = _setup(tmp_path, group=GROUP_T)
    _set_window(c.db, WINDOW_TOO_LONG)
    result = _events(synth.make_force_curve(), c)
    assert result.invols == pytest.approx(DIVISOR_T, rel=1e-12)
    assert len(_ruptures(result)) == 1
    assert _slope_rows(c) == []


def test_an_own_fit_file_whose_fit_fails_still_raises(tmp_path):
    c = _setup(tmp_path)
    _set_window(c.db, WINDOW_TOO_LONG)
    with pytest.raises(curve_analysis.CurveAnalysisError) as raised:
        _events(synth.make_force_curve(), c)
    assert raised.value.stage == "approach invOLS"


@pytest.mark.parametrize("bad", [None, 0.0, -199.14, float("inf")],
                         ids=["null", "zero", "negative", "infinite"])
def test_an_assigned_file_without_a_usable_instrument_divides_by_its_own_slope(
        tmp_path, bad):
    c = _setup(tmp_path, instrument=bad, group=GROUP_T)
    curve = synth.make_force_curve(instrument=bad)
    _analyse(curve, c)
    result = _events(curve, c)
    assert result.invols == _stored_slope(c)


def test_numbers_and_event_map_key_come_from_one_divisor_lookup(
        tmp_path, monkeypatch):
    c = _setup(tmp_path, group=GROUP_T)
    curve = synth.make_force_curve()
    reference = _historical(curve, c)
    real = db.ensemble_divisors
    calls = []

    def lookup_then_reassign(*args, **kwargs):
        result = real(*args, **kwargs)
        if not calls:                     # the assignment moves right after
            synth.add_group(c.db, [c.fid], 220.0)
        calls.append(1)
        return result

    monkeypatch.setattr(db, "ensemble_divisors", lookup_then_reassign)
    _events(curve, c)
    monkeypatch.setattr(db, "ensemble_divisors", real)
    key, payload = synth.fetch(
        c.db, "SELECT params_json, payload_json FROM event_map "
        "WHERE file_id = ?", (c.fid,))[0]
    key_divisor = json.loads(key)["invols_divisor"]
    stored_force = json.loads(payload)["rois"][0]["ruptures"][0]["force_pN"]
    # force is proportional to 1 / |divisor|, so force * |divisor| is the same
    # whichever divisor was used, if the SAME divisor made both.
    assert stored_force * abs(key_divisor) == pytest.approx(
        _forces(reference)[0] * abs(reference.invols), rel=1e-9)


# ── analyse_curve: the own fit is no longer a precondition under an ensemble ──

def test_a_failed_own_fit_without_an_ensemble_raises_as_today(tmp_path):
    c = _setup(tmp_path)
    _set_window(c.db, WINDOW_TOO_LONG)
    with pytest.raises(curve_analysis.CurveAnalysisError) as raised:
        _analyse(synth.make_force_curve(), c)
    assert raised.value.stage == "approach invOLS"


def test_a_failed_own_fit_under_an_ensemble_carries_on(tmp_path):
    c = _setup(tmp_path, group=GROUP_T)
    _set_window(c.db, WINDOW_TOO_LONG)
    result, _ = _analyse(synth.make_force_curve(), c)
    assert math.isnan(result.invols_slope)
    assert result.event is True
    assert _slope_rows(c) == []
    assert _transform_rows(c) == []


def test_a_successful_own_fit_under_an_ensemble_is_still_stored(tmp_path):
    c = _setup(tmp_path, group=GROUP_T)
    result, _ = _analyse(synth.make_force_curve(), c)
    assert result.invols_slope == pytest.approx(OWN_SLOPE, rel=1e-9)
    assert _stored_slope(c) == pytest.approx(OWN_SLOPE, rel=1e-9)


# ── analyse_curve: the own Transform InvOLS row ──────────────────────────────

def test_the_own_transform_row_is_the_instrument_over_the_slope(tmp_path):
    c = _setup(tmp_path)
    _analyse(synth.make_force_curve(), c)
    value, params_json, code_version = _transform_row(c)
    assert value == pytest.approx(209.62105, rel=1e-6)
    assert params_json == synth.current_fit_key(c.db)
    assert code_version == CODE


def test_a_positive_slope_leaves_a_null_transform_row(tmp_path):
    c = _setup(tmp_path)
    _analyse(synth.make_force_curve(slope=0.9), c)
    assert _stored_slope(c) == pytest.approx(0.9, rel=1e-9)
    value, params_json, _ = _transform_row(c)
    assert value is None
    assert params_json == synth.current_fit_key(c.db)


def test_the_own_transform_row_is_refreshed_when_the_fit_comes_from_the_cache(
        tmp_path):
    c = _setup(tmp_path)
    curve = synth.make_force_curve()
    _analyse(curve, c)
    con = sqlite3.connect(c.db)
    with con:
        con.execute("DELETE FROM analysis_results WHERE analysis_type = ?",
                    (TRANSFORM,))
    con.close()
    assert _transform_rows(c) == []
    _analyse(curve, c)
    assert _transform_row(c)[0] == pytest.approx(209.62105, rel=1e-6)


def test_a_new_fit_window_changes_an_own_fit_files_value_in_the_same_pass(
        tmp_path):
    c = _setup(tmp_path)
    curve = synth.make_force_curve(deep_slope=-0.90, deep_pts=200)
    _set_window(c.db, 100)                  # samples n-150 .. n-50: all deep
    _analyse(curve, c)
    deep_value = _transform_row(c)[0]
    assert deep_value == pytest.approx(INSTRUMENT / 0.90, rel=1e-6)
    _set_window(c.db, 250)                  # samples n-300 .. n-50: both parts
    _analyse(curve, c)
    value, params_json, _ = _transform_row(c)
    assert INSTRUMENT / 0.95 < value < deep_value
    assert params_json == synth.current_fit_key(c.db)


def test_an_assigned_file_keeps_its_applied_row_through_a_rerun(tmp_path):
    c = _setup(tmp_path)
    group_id = synth.add_group(c.db, [c.fid], GROUP_T)
    _seed_applied_row(c, group_id)
    before = _transform_rows(c)
    curve = synth.make_force_curve(deep_slope=-0.90, deep_pts=200)
    _analyse(curve, c)
    assert _transform_rows(c) == before
    _set_window(c.db, 100)
    _analyse(curve, c)
    assert _transform_rows(c) == before
    assert before[0][2] == "APPLIED"       # code_version and computed_at too


def test_an_assigned_file_with_no_row_gets_no_row_from_a_run(tmp_path):
    c = _setup(tmp_path, group=GROUP_T)
    _analyse(synth.make_force_curve(), c)
    assert _transform_rows(c) == []


# ── analyse_curve: an assigned file with no usable instrument is an own-fit file

UNUSABLE = [None, 0.0, -199.14, float("inf")]
UNUSABLE_IDS = ["null", "zero", "negative", "infinite"]


@pytest.mark.parametrize("bad", UNUSABLE, ids=UNUSABLE_IDS)
def test_such_a_file_whose_fit_fails_raises_and_its_row_becomes_null(
        tmp_path, bad):
    c = _setup(tmp_path, instrument=bad)
    group_id = synth.add_group(c.db, [c.fid], GROUP_T)
    _seed_applied_row(c, group_id)
    _set_window(c.db, WINDOW_TOO_LONG)
    with pytest.raises(curve_analysis.CurveAnalysisError) as raised:
        _analyse(synth.make_force_curve(instrument=bad), c)
    assert raised.value.stage == "approach invOLS"
    value, params_json, code_version = _transform_row(c)
    assert value is None
    assert params_json == synth.current_fit_key(c.db)
    assert code_version == CODE


@pytest.mark.parametrize("bad", UNUSABLE, ids=UNUSABLE_IDS)
def test_such_a_file_whose_fit_succeeds_has_its_row_set_to_null(tmp_path, bad):
    c = _setup(tmp_path, instrument=bad)
    group_id = synth.add_group(c.db, [c.fid], GROUP_T)
    _seed_applied_row(c, group_id)
    result, _ = _analyse(synth.make_force_curve(instrument=bad), c)
    assert result.invols_slope == pytest.approx(OWN_SLOPE, rel=1e-9)
    value, params_json, code_version = _transform_row(c)
    assert value is None
    assert params_json == synth.current_fit_key(c.db)
    assert code_version == CODE


# ── the 2DH window is not part of this feature ───────────────────────────────

def test_base_2dh_window_is_unchanged_from_main():
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    probe = subprocess.run(
        [git, "rev-parse", "--is-inside-work-tree", "main"],
        cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("no git checkout with a main branch here")
    diff = subprocess.run(
        [git, "diff", "main", "--", "smfs_catalog/base_2dh_window.py"],
        cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0
    assert diff.stdout == ""
