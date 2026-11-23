# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The shared invOLS fit controls (offset and window spin boxes) and the curve
view's "InvOLS in use" label.

The pair of spin boxes lives in one widget, InvolsFitControls, used by the
curve view and (later) the invOLS window.  A user edit writes the catalog and
the active owner's profile and emits `changed` once; set_values and
refresh_from_db only move what is shown and emit and write nothing.

The curve view's label comes from invols_store.curve_label, drawn through the
real window on a synthetic curve (tests/invols_synth.py: noiseless contact of
a known slope).  Hand calculation: instrument 199.14 nm/V with slope -0.9852
gives 199.14 / 0.9852 = 202.1315 -> "202.13".  (The plan's "202.12" is only a
format example.)  Every window is given an explicit temporary catalog and the
code version is pinned, so no git checkout is needed.
"""
import dataclasses
import os
import re
import sqlite3
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import invols_synth as synth                                  # noqa: E402
from PyQt6.QtWidgets import QApplication, QSpinBox            # noqa: E402
from smfs_catalog import (curve_analysis, db, decomposition_window as dw,  # noqa: E402
                          invols_store, roi_pipeline)

_PKG = Path(__file__).resolve().parents[1] / "smfs_catalog"
INSTRUMENT = 199.14
OWNER = "Ada"
WINDOW_TOO_LONG = 5000        # the synthetic approach has 3000 samples


@pytest.fixture(autouse=True)
def pinned_version(monkeypatch):
    """The code version every module stamps and checks, fixed so these tests
    need no git checkout (invols_synth writes its rows under the same text)."""
    for module in (curve_analysis, roi_pipeline, invols_store, dw):
        monkeypatch.setattr(module, "cache_version", lambda: synth.CODE)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def catalog(tmp_path):
    """Three files, instrument 199.14 each, the first (the queue's head) owned
    by OWNER so the active parameter owner is a named person."""
    db_path, ids = synth.make_catalog(tmp_path, [
        {"instrument": INSTRUMENT, "measured_at": "2026-01-01 10:00:00"},
        {"instrument": INSTRUMENT, "measured_at": "2026-01-01 11:00:00"},
        {"instrument": INSTRUMENT, "measured_at": "2026-01-01 12:00:00"},
    ])
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("UPDATE files SET experimentalist = ? WHERE id = ?",
                        (OWNER, ids[0]))
    finally:
        con.close()
    return db_path, ids


@pytest.fixture
def open_windows(qapp):
    made = []
    yield made
    for widget in made:
        widget.close()


def _controls(db_path, made):
    from smfs_catalog.invols_fit_controls import InvolsFitControls
    widget = InvolsFitControls(db_path)
    made.append(widget)
    return widget


def _window(db_path, made):
    win = dw.DecompositionWindow(db_path)
    made.append(win)
    return win


def _counter(signal):
    box = []
    signal.connect(lambda *args: box.append(args))
    return box


def _profile(db_path):
    return db.get_experimentalist_profile(db.active_param_owner(db_path), db_path)


def _stored(db_path):
    params = db.load_analysis_params(db_path)
    return params.invols_offset_pts, params.invols_window_pts


def _curve(file_number=1, slope=-0.9852):
    """A synthetic curve whose path is the catalog path of file `file_number`."""
    base = synth.make_force_curve(slope=slope, instrument=INSTRUMENT)
    return dataclasses.replace(base, path=synth.path_of(file_number))


def _shown(win, curve):
    win.update_curve(curve)
    return win._invols_label.text()


def _fit_line_points(win):
    xs, _ys = win._invols_fit_line.getData()
    return 0 if xs is None else len(xs)


# ── the shared widget on its own ─────────────────────────────────────────────

def test_defaults_show_offset_50_and_window_200(catalog, open_windows):
    controls = _controls(catalog[0], open_windows)
    assert controls.offset_pts() == 50
    assert controls.window_pts() == 200
    assert controls.offset_spinbox.value() == 50
    assert controls.window_spinbox.value() == 200


def test_spin_box_floors_are_0_and_2(catalog, open_windows):
    controls = _controls(catalog[0], open_windows)
    assert isinstance(controls.offset_spinbox, QSpinBox)
    assert isinstance(controls.window_spinbox, QSpinBox)
    assert controls.offset_spinbox.minimum() == 0
    assert controls.window_spinbox.minimum() == 2


def test_labels_are_in_the_shared_widget(catalog, open_windows):
    from PyQt6.QtWidgets import QLabel
    controls = _controls(catalog[0], open_windows)
    texts = [label.text() for label in controls.findChildren(QLabel)]
    assert "invOLS off:" in texts
    assert "invOLS win:" in texts


def test_user_edit_reaches_catalog_and_owner_profile_and_emits_once(
        catalog, open_windows):
    db_path, _ids = catalog
    controls = _controls(db_path, open_windows)
    emitted = _counter(controls.changed)
    controls.offset_spinbox.setValue(70)
    assert db.load_analysis_params(db_path).invols_offset_pts == 70
    assert db.active_param_owner(db_path) == OWNER
    assert db.get_experimentalist_profile(OWNER, db_path)["invols_offset_pts"] == 70.0
    assert len(emitted) == 1


def test_window_edit_reaches_catalog_and_owner_profile_and_emits_once(
        catalog, open_windows):
    db_path, _ids = catalog
    controls = _controls(db_path, open_windows)
    emitted = _counter(controls.changed)
    controls.window_spinbox.setValue(120)
    assert db.load_analysis_params(db_path).invols_window_pts == 120
    assert db.get_experimentalist_profile(OWNER, db_path)["invols_window_pts"] == 120.0
    assert len(emitted) == 1


def test_user_edit_keeps_the_other_boxes_stored_value(catalog, open_windows):
    db_path, _ids = catalog
    controls = _controls(db_path, open_windows)
    controls.offset_spinbox.setValue(70)
    assert _stored(db_path) == (70, 200)
    assert controls.window_pts() == 200


def test_refresh_from_db_shows_outside_change_and_emits_nothing(
        catalog, open_windows):
    db_path, _ids = catalog
    controls = _controls(db_path, open_windows)
    emitted = _counter(controls.changed)
    db.update_analysis_param("invols_window_pts", 120, db_path)
    before = _profile(db_path)
    controls.refresh_from_db()
    assert controls.window_spinbox.value() == 120
    assert controls.window_pts() == 120
    assert controls.offset_pts() == 50
    assert emitted == []
    assert _stored(db_path) == (50, 120)
    assert _profile(db_path) == before


def test_set_values_moves_the_boxes_without_emitting_or_writing(
        catalog, open_windows):
    db_path, _ids = catalog
    controls = _controls(db_path, open_windows)
    emitted = _counter(controls.changed)
    stored_before, profile_before = _stored(db_path), _profile(db_path)
    controls.set_values(80, 90)
    assert (controls.offset_pts(), controls.window_pts()) == (80, 90)
    assert (controls.offset_spinbox.value(), controls.window_spinbox.value()) == (80, 90)
    assert emitted == []
    assert _stored(db_path) == stored_before == (50, 200)
    assert _profile(db_path) == profile_before


def test_a_second_controls_agrees_after_refresh(catalog, open_windows):
    db_path, _ids = catalog
    first = _controls(db_path, open_windows)
    second = _controls(db_path, open_windows)
    first.offset_spinbox.setValue(33)
    first.window_spinbox.setValue(44)
    assert (second.offset_pts(), second.window_pts()) == (50, 200)
    second.refresh_from_db()
    assert (second.offset_pts(), second.window_pts()) == (33, 44)


# ── the curve view uses the shared widget ────────────────────────────────────

def test_curve_view_spin_boxes_are_the_shared_widgets(catalog, open_windows):
    from smfs_catalog.invols_fit_controls import InvolsFitControls
    win = _window(catalog[0], open_windows)
    shared = win.findChildren(InvolsFitControls)
    assert len(shared) == 1
    assert win._invols_offset_spinbox is shared[0].offset_spinbox
    assert win._invols_window_spinbox is shared[0].window_spinbox


def test_decomposition_window_source_builds_no_invols_spin_box():
    """The boxes MOVE: the window no longer constructs its own invOLS box,
    tooltip or label text."""
    source = (_PKG / "decomposition_window.py").read_text(encoding="utf-8")
    shared = (_PKG / "invols_fit_controls.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert not ("invols" in line.lower() and "QSpinBox(" in line), line
    assert "invOLS off:" not in source and "invOLS win:" not in source
    assert "invOLS off:" in shared and "invOLS win:" in shared
    assert re.search(r"\bQSpinBox\(", shared)
    assert "InvolsFitControls" in source


def test_curve_view_edit_emits_analysis_params_changed_once(
        catalog, open_windows):
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    emitted = _counter(win.analysis_params_changed)
    win._invols_offset_spinbox.setValue(70)
    assert len(emitted) == 1
    assert db.load_analysis_params(db_path).invols_offset_pts == 70
    assert db.get_experimentalist_profile(OWNER, db_path)["invols_offset_pts"] == 70.0
    assert win._invols_offset_pts == 70


def test_curve_view_window_edit_emits_analysis_params_changed_once(
        catalog, open_windows):
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    emitted = _counter(win.analysis_params_changed)
    win._invols_window_spinbox.setValue(150)
    assert len(emitted) == 1
    assert db.load_analysis_params(db_path).invols_window_pts == 150
    assert win._invols_window_pts == 150


def test_curve_view_edit_redraws_the_current_curve(catalog, open_windows):
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    win.update_curve(_curve())
    assert _fit_line_points(win) == 2
    win._invols_window_spinbox.setValue(WINDOW_TOO_LONG)
    assert _fit_line_points(win) == 0


# ── the curve view's label ───────────────────────────────────────────────────

def test_label_of_an_own_fit_trace(catalog, open_windows):
    win = _window(catalog[0], open_windows)
    text = _shown(win, _curve())
    assert "InvOLS in use 202.13 nm/V (own fit)" in text
    assert "slope=-0.9852" in text
    assert "RMS=" in text
    assert "ensemble" not in text
    assert "invOLS slope=" not in text


def test_label_of_an_ensemble_trace_with_an_own_fit(catalog, open_windows):
    db_path, ids = catalog
    synth.add_group(db_path, ids, 210.0)
    win = _window(db_path, open_windows)
    text = _shown(win, _curve())
    assert "InvOLS in use 210.00 nm/V (ensemble of 3 traces)" in text
    assert "own fit 202.13 nm/V" in text
    assert "slope=-0.9852" in text
    assert "RMS=" in text


def test_label_of_an_ensemble_trace_with_no_own_fit(catalog, open_windows):
    db_path, ids = catalog
    synth.add_group(db_path, ids, 210.0)
    db.update_analysis_param("invols_window_pts", WINDOW_TOO_LONG, db_path)
    win = _window(db_path, open_windows)
    text = _shown(win, _curve())
    assert "InvOLS in use 210.00 nm/V (ensemble of 3 traces)" in text
    assert "own fit —" in text
    assert "slope=" not in text
    assert "RMS=" not in text


def test_ensemble_label_follows_an_edit_to_a_window_with_no_own_fit(
        catalog, open_windows):
    db_path, ids = catalog
    synth.add_group(db_path, ids, 210.0)
    win = _window(db_path, open_windows)
    assert "own fit 202.13 nm/V" in _shown(win, _curve())
    win._invols_window_spinbox.setValue(WINDOW_TOO_LONG)
    assert "own fit —" in win._invols_label.text()
    assert "(ensemble of 3 traces)" in win._invols_label.text()


def test_label_of_an_assigned_file_with_an_unusable_instrument_is_own_fit(
        catalog, open_windows):
    """No ensemble is in force when the file's instrument value is unusable,
    whatever its assignment row says, so the trace reads as an own-fit trace."""
    db_path, ids = catalog
    synth.add_group(db_path, ids, 210.0)
    synth.set_instrument(db_path, ids[0], 0.0)
    win = _window(db_path, open_windows)
    text = _shown(win, _curve())
    assert "(own fit)" in text
    assert "ensemble" not in text
    assert "210.00" not in text


def test_fit_line_is_still_drawn_for_an_own_fit(catalog, open_windows):
    win = _window(catalog[0], open_windows)
    win.update_curve(_curve())
    assert _fit_line_points(win) == 2


def test_fit_line_is_still_drawn_under_an_ensemble_with_an_own_fit(
        catalog, open_windows):
    db_path, ids = catalog
    synth.add_group(db_path, ids, 210.0)
    win = _window(db_path, open_windows)
    win.update_curve(_curve())
    assert _fit_line_points(win) == 2


def test_no_fit_line_when_there_is_no_own_fit(catalog, open_windows):
    db_path, ids = catalog
    synth.add_group(db_path, ids, 210.0)
    db.update_analysis_param("invols_window_pts", WINDOW_TOO_LONG, db_path)
    win = _window(db_path, open_windows)
    win.update_curve(_curve())
    assert _fit_line_points(win) == 0


# ── hover text: the wording that moved from decomposition_window.py ──────────
# Literals copied from `main`'s decomposition_window.py, so a silent wording
# change during the move fails.  Read from the spin boxes themselves (the old
# bug was tooltips set on status labels instead of on the control).

_OFFSET_TIP = (
    "Samples skipped back from the turnaround before the invOLS fit "
    "window begins. The deepest-contact samples carry piezo ringing and "
    "any sample deformation at peak load, both of which bend the line "
    "that converts deflection into force.\n\n"
    "If the offset and window together reach past the start of the "
    "approach, no slope can be fitted and the curve is not analysed: "
    "its readouts show \"—\".")
_WINDOW_TIP = (
    "Number of samples in the invOLS straight-line fit, counted back "
    "from the offset above toward shallower contact.\n\n"
    "This slope is what turns raw deflection into force, so it scales "
    "every force in the app. The fitted slope and RMS residual are shown "
    "beside it. If the window reaches past the start of the approach, "
    "no slope can be fitted and the curve is not analysed.")


def test_spin_boxes_carry_the_original_hover_text(catalog, open_windows):
    controls = _controls(catalog[0], open_windows)
    assert controls.offset_spinbox.toolTip() == _OFFSET_TIP
    assert controls.window_spinbox.toolTip() == _WINDOW_TIP


# ── a second control's edit must not be undone by the curve view ─────────────
# Another InvolsFitControls on the same catalog sets the offset 50 -> 70 while
# the curve view's own box still holds 50.

OTHER_OFFSET = 70
_WINDOW_KEYS = ("spectral_cutoff_hz", "turnaround_trim_pts", "var_window_ms",
                "detection_threshold_appr", "detection_threshold_retr",
                "baseline_anchor_nm", "invols_offset_pts", "invols_window_pts")


@pytest.fixture
def outside_edit(catalog, open_windows):
    """(db_path, window, key of the fit under offset 50, key under offset 70):
    the window is open, then a second shared widget sets the offset to 70."""
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    key_before = synth.current_fit_key(db_path)
    _controls(db_path, open_windows).offset_spinbox.setValue(OTHER_OFFSET)
    key_after = synth.current_fit_key(db_path)
    assert key_before != key_after            # the test can tell them apart
    assert win._invols_offset_spinbox.value() == 50
    return db_path, win, key_before, key_after


def _owner_profile(db_path):
    return db.get_experimentalist_profile(OWNER, db_path)


def _slope_key(db_path):
    rows = synth.fetch(db_path, "SELECT params_json FROM analysis_results "
                       "WHERE file_id = 1 AND analysis_type = 'invols_slope'")
    return rows[0][0] if rows else None


def test_editing_the_curve_views_window_box_keeps_the_others_offset(
        outside_edit):
    db_path, win, _k50, _k70 = outside_edit
    win._invols_window_spinbox.setValue(120)
    assert _stored(db_path) == (OTHER_OFFSET, 120)
    profile = _owner_profile(db_path)
    assert profile["invols_offset_pts"] == float(OTHER_OFFSET)
    assert profile["invols_window_pts"] == 120.0


def test_editing_another_curve_view_control_keeps_the_others_offset(
        outside_edit):
    db_path, win, _k50, _k70 = outside_edit
    win._trim_spinbox.setValue(7)
    assert _stored(db_path)[0] == OTHER_OFFSET
    assert _owner_profile(db_path)["invols_offset_pts"] == float(OTHER_OFFSET)
    win._cutoff_slider.setValue(win._cutoff_slider.value() - 1)
    assert _stored(db_path)[0] == OTHER_OFFSET
    assert _owner_profile(db_path)["invols_offset_pts"] == float(OTHER_OFFSET)


def test_a_plain_redraw_fits_under_the_others_offset(outside_edit):
    db_path, win, key_50, key_70 = outside_edit
    win.update_curve(_curve())
    assert _slope_key(db_path) == key_70
    assert _slope_key(db_path) != key_50
    assert win._invols_offset_spinbox.value() == OTHER_OFFSET
    assert win._invols_offset_pts == OTHER_OFFSET
    assert _stored(db_path)[0] == OTHER_OFFSET


def test_refresh_invols_fit_adopts_the_outside_edit_and_redraws_once(
        outside_edit):
    db_path, win, _k50, key_70 = outside_edit
    win.update_curve(_curve())          # shown, under whatever is in force
    emitted = _counter(win.analysis_params_changed)
    params_before, profile_before = db.load_analysis_params(db_path), _owner_profile(db_path)
    win.refresh_invols_fit()
    assert win._invols_offset_spinbox.value() == OTHER_OFFSET
    assert win._invols_offset_pts == OTHER_OFFSET
    assert win._invols_window_pts == 200
    assert len(emitted) == 1
    assert _slope_key(db_path) == key_70                 # the curve was redrawn
    assert db.load_analysis_params(db_path) == params_before
    assert _owner_profile(db_path) == profile_before


def test_refresh_invols_fit_with_no_curve_showing_does_not_raise(
        outside_edit):
    db_path, win, _k50, _k70 = outside_edit
    win.refresh_invols_fit()
    assert win._invols_offset_spinbox.value() == OTHER_OFFSET
    assert win._invols_offset_pts == OTHER_OFFSET
    assert _stored(db_path) == (OTHER_OFFSET, 200)


def test_a_brand_new_owner_gets_a_complete_profile_with_the_invols_keys(
        catalog, open_windows):
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    before = set(_owner_profile(db_path) or {})
    win._sync_profile_owner(synth.path_of(1))
    profile = _owner_profile(db_path)
    params = db.load_analysis_params(db_path)
    assert before <= set(profile)
    for key in _WINDOW_KEYS:
        assert key in profile, key
    assert profile["invols_offset_pts"] == float(params.invols_offset_pts) == 50.0
    assert profile["invols_window_pts"] == float(params.invols_window_pts) == 200.0


# ── a failed fit must not leave an old RMS on the label ──────────────────────

def test_own_fit_label_drops_the_rms_when_the_fit_no_longer_exists(
        catalog, open_windows):
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    assert "RMS=" in _shown(win, _curve())              # a fit, and its RMS
    win._invols_window_spinbox.setValue(WINDOW_TOO_LONG)
    text = win._invols_label.text()
    assert "RMS=" not in text
    assert "(own fit)" in text


def test_ensemble_label_drops_the_rms_when_the_fit_no_longer_exists(
        catalog, open_windows):
    db_path, ids = catalog
    synth.add_group(db_path, ids, 210.0)
    win = _window(db_path, open_windows)
    assert "RMS=" in _shown(win, _curve())
    win._invols_window_spinbox.setValue(WINDOW_TOO_LONG)
    text = win._invols_label.text()
    assert "own fit —" in text
    assert "RMS=" not in text


# ── smaller items ────────────────────────────────────────────────────────────

def test_label_tooltip_holds_the_full_text(catalog, open_windows):
    win = _window(catalog[0], open_windows)
    text = _shown(win, _curve())
    assert text
    assert win._invols_label.toolTip() == text


def test_a_failing_label_does_not_crash_the_redraw(
        catalog, open_windows, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("label failed")
    monkeypatch.setattr(invols_store, "curve_label", boom)
    win = _window(catalog[0], open_windows)
    win.update_curve(_curve())                      # must return
    xs, _ys = win._low_appr.getData()
    assert xs is not None and len(xs) > 0


def test_a_failing_label_clears_text_and_tooltip_after_a_good_one(
        catalog, open_windows, monkeypatch):
    win = _window(catalog[0], open_windows)
    assert "InvOLS in use" in _shown(win, _curve())     # a good label first
    assert win._invols_label.toolTip()

    def boom(*_args, **_kwargs):
        raise RuntimeError("label failed")
    monkeypatch.setattr(invols_store, "curve_label", boom)
    win.update_curve(_curve())
    assert win._invols_label.text() == ""
    assert win._invols_label.toolTip() == ""


# ── a redraw must not discard a value still being typed ──────────────────────
# Keyboard tracking is off, so typed digits stay uncommitted until Enter.

def _type_without_enter(win, box, digits):
    from PyQt6.QtTest import QTest
    win.show()
    box.setFocus()
    box.selectAll()
    QTest.keyClicks(box.lineEdit(), digits)


def _press_enter(box):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    QTest.keyClick(box, Qt.Key.Key_Return)


def test_typing_in_the_window_box_survives_a_redraw(catalog, open_windows):
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    win.update_curve(_curve())
    box = win._invols_window_spinbox
    _type_without_enter(win, box, "150")
    win.update_curve(_curve())                    # a queue step, say
    _press_enter(box)
    assert box.value() == 150
    assert _stored(db_path)[1] == 150


def test_typing_in_the_offset_box_survives_a_redraw(catalog, open_windows):
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    win.update_curve(_curve())
    box = win._invols_offset_spinbox
    _type_without_enter(win, box, "30")
    win.update_curve(_curve())
    _press_enter(box)
    assert box.value() == 30
    assert _stored(db_path)[0] == 30


def test_a_redraw_adopts_a_catalog_value_when_the_user_is_not_typing(
        catalog, open_windows):
    db_path, _ids = catalog
    win = _window(db_path, open_windows)
    win.update_curve(_curve())
    _controls(db_path, open_windows).window_spinbox.setValue(120)
    assert win._invols_window_spinbox.value() == 200
    win.update_curve(_curve())
    assert win._invols_window_spinbox.value() == 120
    assert win._invols_window_pts == 120


# ── the RMS of a genuine fit shows even when nothing is stored ───────────────

def _no_code_identity(monkeypatch):
    from smfs_catalog import provenance
    for module in (curve_analysis, roi_pipeline, invols_store, dw, provenance):
        monkeypatch.setattr(module, "cache_version", lambda: None)


def test_rms_shows_for_a_good_fit_when_the_build_has_no_code_identity(
        catalog, open_windows, monkeypatch):
    db_path, _ids = catalog
    _no_code_identity(monkeypatch)
    win = _window(db_path, open_windows)
    text = _shown(win, _curve())
    assert "slope=-0.9852" in text
    assert "RMS=" in text
    assert synth.fetch(db_path, "SELECT COUNT(*) FROM analysis_results WHERE "
                       "analysis_type = 'invols_slope'") == [(0,)]


def test_rms_shows_for_a_good_fit_of_a_curve_with_no_catalog_row(
        catalog, open_windows):
    win = _window(catalog[0], open_windows)
    curve = dataclasses.replace(_curve(), path=synth.path_of(99))
    text = _shown(win, curve)
    assert "slope=-0.9852" in text
    assert "RMS=" in text


# ── two owners with different invOLS values ──────────────────────────────────

def test_switching_the_queue_to_another_owner_adopts_their_invols_values(
        tmp_path, open_windows):
    db_path, ids = synth.make_catalog(tmp_path, [
        {"instrument": INSTRUMENT, "measured_at": "2026-01-01 10:00:00"},
        {"instrument": INSTRUMENT, "measured_at": "2026-01-01 11:00:00"},
    ])
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("UPDATE files SET experimentalist = 'Ada' WHERE id = ?", (ids[0],))
            con.execute("UPDATE files SET experimentalist = 'Ben' WHERE id = ?", (ids[1],))
    finally:
        con.close()
    db.merge_experimentalist_profile(
        "Ada", {"invols_offset_pts": 40.0, "invols_window_pts": 180.0}, db_path)
    db.merge_experimentalist_profile(
        "Ben", {"invols_offset_pts": 60.0, "invols_window_pts": 220.0}, db_path)
    win = _window(db_path, open_windows)
    win._sync_profile_owner(synth.path_of(1))
    win.update_curve(_curve(1))
    assert (win._invols_offset_pts, win._invols_window_pts) == (40, 180)

    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("DELETE FROM analysis_queue WHERE file_id = ?", (ids[0],))
    finally:
        con.close()
    assert db.active_param_owner(db_path) == "Ben"
    win._sync_profile_owner(synth.path_of(2))
    win.update_curve(_curve(2))

    assert (win._invols_offset_pts, win._invols_window_pts) == (60, 220)
    assert (win._invols_offset_spinbox.value(),
            win._invols_window_spinbox.value()) == (60, 220)
    rows = synth.fetch(db_path, "SELECT params_json FROM analysis_results "
                       "WHERE file_id = 2 AND analysis_type = 'invols_slope'")
    assert rows == [(synth.current_fit_key(db_path),)]
    ada = db.get_experimentalist_profile("Ada", db_path)
    ben = db.get_experimentalist_profile("Ben", db_path)
    assert (ada["invols_offset_pts"], ada["invols_window_pts"]) == (40.0, 180.0)
    assert (ben["invols_offset_pts"], ben["invols_window_pts"]) == (60.0, 220.0)
