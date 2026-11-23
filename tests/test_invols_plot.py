# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
The invOLS plot: one point per trace, time on x and nm/V on y, with a box drawn
by a left drag.

The widget needs no catalog, so the rows are built by hand.  Point state is
read the way pyqtgraph itself exposes it, so the tests do not depend on how the
widget arranges its items:

* every ``ScatterPlotItem`` added to the plot's view box is searched; each of
  its spots gives ``pos()``, ``pen()``, ``brush()``, ``symbol()`` and ``data()``;
* a pinned-edge marker is a spot whose symbol is a triangle;
* a tooltip is what the scatter's ``opts['tip']`` callable returns for the spot
  (the same call pyqtgraph makes when the pointer hovers it);
* clicks, double clicks and drags are real mouse events sent to the viewport.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pg = pytest.importorskip("pyqtgraph")
pytest.importorskip("PyQt6")

import numpy as np                                                  # noqa: E402
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt                # noqa: E402
from PyQt6.QtGui import QMouseEvent                                 # noqa: E402
from PyQt6.QtTest import QTest                                      # noqa: E402
from PyQt6.QtWidgets import QApplication                            # noqa: E402

from smfs_catalog import quantities, style                          # noqa: E402
from smfs_catalog.invols_store import TraceRow                      # noqa: E402
from smfs_catalog.qt_utils import _DateAxis                         # noqa: E402

T0 = 1_700_000_000.0          # an arbitrary Unix time; rows are an hour apart
HOUR = 3600.0
TRIANGLES = {"t", "t1", "t2", "t3"}
LEFT, RIGHT, MIDDLE = (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton,
                       Qt.MouseButton.MiddleButton)


# ── builders and readers ─────────────────────────────────────────────────────

def _row(i, own, *, source="own", current=True, time_s="auto"):
    t = T0 + i * HOUR if time_s == "auto" else time_s
    return TraceRow(file_id=i + 1, path=f"/data/f{i + 1}.ibw", time_s=t,
                    instrument_nm_v=199.14, own_nm_v=own, fit_current=current,
                    rms_nm=0.2, source=source,
                    group_id=None if source == "own" else 1,
                    in_use_nm_v=own)


def _rows(values, **kw):
    return [_row(i, v, **kw) for i, v in enumerate(values)]


def _scatters(plot):
    return [it for it in plot.getViewBox().addedItems
            if isinstance(it, pg.ScatterPlotItem)]


def _spots(plot):
    return [(sc, pt) for sc in _scatters(plot) for pt in sc.points()]


def _points(plot):
    """Drawn spots that are not pinned-edge markers."""
    return [(sc, pt) for sc, pt in _spots(plot) if pt.symbol() not in TRIANGLES]


def _markers(plot):
    return [(sc, pt) for sc, pt in _spots(plot) if pt.symbol() in TRIANGLES]


def _tip(scatter, spot) -> str:
    tip = scatter.opts["tip"]
    assert callable(tip), "the scatter has no tooltip callable (opts['tip'])"
    return str(tip(x=spot.pos().x(), y=spot.pos().y(), data=spot.data()))


def _tips(plot):
    return [_tip(sc, pt) for sc, pt in _spots(plot)]


def _is_filled(spot) -> bool:
    brush = spot.brush()
    return (brush.style() != Qt.BrushStyle.NoBrush
            and brush.color().alpha() > 0)


def _pen_hex(spot) -> str:
    return spot.pen().color().name().lower()


def _at(plot, t, y) -> QPoint:
    scene = plot.getViewBox().mapViewToScene(QPointF(t, y))
    return plot.mapFromScene(scene)


def _view_y(plot):
    return plot.getViewBox().viewRange()[1]


def _horizontal_lines(plot, value):
    """Visible horizontal lines at ``value`` (an InfiniteLine or a flat curve)."""
    found = []
    for it in plot.getViewBox().addedItems:
        if not it.isVisible():
            continue
        if isinstance(it, pg.InfiniteLine) and it.angle == 0:
            if math.isclose(float(it.value()), value, rel_tol=1e-9):
                found.append(it)
        elif isinstance(it, (pg.PlotDataItem, pg.PlotCurveItem)):
            y = it.getData()[1]
            if y is not None and len(y) and np.allclose(y, value, rtol=1e-9):
                found.append(it)
    return found


def _send(plot, kind, pos, button, buttons):
    vp = plot.viewport()
    ev = QMouseEvent(kind, QPointF(pos), QPointF(vp.mapToGlobal(pos)), button,
                     buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(vp, ev)
    QApplication.processEvents()


def _drag(plot, start: QPoint, end: QPoint, button=LEFT):
    """A press, a few moves and a release.  The first move is a few pixels
    from the press, as a real hand's is."""
    _send(plot, QEvent.Type.MouseButtonPress, start, button, button)
    dx, dy = end.x() - start.x(), end.y() - start.y()
    for frac in (0.03, 0.5, 1.0):
        pos = QPoint(start.x() + int(dx * frac), start.y() + int(dy * frac))
        _send(plot, QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, button)
    _send(plot, QEvent.Type.MouseButtonRelease, end, button,
          Qt.MouseButton.NoButton)


class _Spy:
    def __init__(self, signal):
        self.calls = []
        signal.connect(lambda *a: self.calls.append(tuple(a)))


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def plot(qapp):
    from smfs_catalog.invols_plot import InvolsPlot
    w = InvolsPlot()
    w.resize(640, 420)
    w.show()
    QApplication.processEvents()
    yield w
    w.close()
    w.deleteLater()


def _settle(plot):
    QApplication.processEvents()


# ── points, colours, hollow ──────────────────────────────────────────────────

def test_three_rows_with_own_values_give_three_points(plot):
    plot.set_traces(_rows([209.62105, 199.14, 189.65714]))
    _settle(plot)
    assert len(_points(plot)) == 3
    assert not _markers(plot)
    xs = sorted(pt.pos().x() for _, pt in _points(plot))
    assert xs == pytest.approx([T0, T0 + HOUR, T0 + 2 * HOUR])
    ys = sorted(pt.pos().y() for _, pt in _points(plot))
    assert ys == pytest.approx([189.65714, 199.14, 209.62105])


def test_rows_with_no_own_value_add_no_point(plot):
    plot.set_traces(_rows([209.0, None, 199.0, None]))
    _settle(plot)
    assert len(_spots(plot)) == 2


def test_axes_are_time_and_nm_per_v(plot):
    assert isinstance(plot.getAxis("bottom"), _DateAxis)
    left = plot.getAxis("left")
    label = f"{left.labelText} {left.labelUnits}"
    assert quantities.unit_of("transform_invols_nm_v") in label


def test_own_and_ensemble_use_two_different_named_style_colours(plot):
    plot.set_traces([_row(0, 200.0, source="own"),
                     _row(1, 201.0, source="ensemble"),
                     _row(2, 199.0, source="own")])
    _settle(plot)
    by_x = {round(pt.pos().x()): _pen_hex(pt) for _, pt in _points(plot)}
    own, ens = by_x[round(T0)], by_x[round(T0 + HOUR)]
    assert by_x[round(T0 + 2 * HOUR)] == own
    assert own != ens
    palette = {c.lower() for c in (*style.SERIES_LINE, *style.SERIES_LABELED)}
    palette |= {c.lower() for c in (style.INK, style.INK_STRONG, style.INK_MUTED,
                                    style.INK_FAINT, style.DATA, style.GRID)}
    for colour in (own, ens):
        named = [n for n, v in vars(style).items()
                 if isinstance(v, str) and v.lower() == colour]
        assert named, f"{colour} is not a named style.py token"
        assert colour in palette, f"{colour} is not from the existing palette"


def test_plot_module_names_no_colour_literal():
    import smfs_catalog.invols_plot as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert not re.search(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?\b", src)
    assert not re.search(r"QColor\(\s*\d", src)


def test_fit_current_false_is_hollow_and_true_is_filled(plot):
    plot.set_traces([_row(0, 200.0, current=True),
                     _row(1, 201.0, current=False)])
    _settle(plot)
    spots = {round(pt.pos().x()): (sc, pt) for sc, pt in _points(plot)}
    filled = spots[round(T0)][1]
    hollow = spots[round(T0 + HOUR)][1]
    assert _is_filled(filled)
    assert not _is_filled(hollow)
    assert hollow.pen().color().alpha() > 0      # an outline is still drawn


def test_hollow_tooltip_says_earlier_fit_settings_and_filled_does_not(plot):
    plot.set_traces([_row(0, 200.0, current=True),
                     _row(1, 201.0, current=False)])
    _settle(plot)
    tips = {round(pt.pos().x()): _tip(sc, pt) for sc, pt in _points(plot)}
    assert "earlier fit settings" in tips[round(T0 + HOUR)]
    assert "earlier fit settings" not in tips[round(T0)]


# ── the box ──────────────────────────────────────────────────────────────────

def test_commit_box_normalises_stores_and_emits_once(plot):
    spy = _Spy(plot.box_drawn)
    plot.commit_box(5, 1, 300, 100)
    assert spy.calls == [(1.0, 5.0, 100.0, 300.0)]
    assert plot.box() == (1.0, 5.0, 100.0, 300.0)


def test_clear_box_gives_none(plot):
    plot.commit_box(5, 1, 300, 100)
    plot.clear_box()
    assert plot.box() is None


def test_no_box_before_one_is_drawn(plot):
    assert plot.box() is None


def _prepare_for_drag(plot):
    plot.set_traces(_rows([200.0, 201.0, 199.0, 200.5]))
    plot.setXRange(T0, T0 + 4 * HOUR, padding=0)
    plot.setYRange(0, 40, padding=0)
    _settle(plot)


def test_left_drag_draws_a_box_in_data_coordinates(plot):
    _prepare_for_drag(plot)
    spy = _Spy(plot.box_drawn)
    # from the upper right to the lower left, so normalising is exercised too
    _drag(plot, _at(plot, T0 + 3.5 * HOUR, 35), _at(plot, T0 + 0.5 * HOUR, 5))
    assert len(spy.calls) == 1
    t0, t1, y0, y1 = spy.calls[0]
    span_t, span_y = 4 * HOUR, 40
    assert t0 == pytest.approx(T0 + 0.5 * HOUR, abs=0.02 * span_t)
    assert t1 == pytest.approx(T0 + 3.5 * HOUR, abs=0.02 * span_t)
    assert y0 == pytest.approx(5, abs=0.02 * span_y)
    assert y1 == pytest.approx(35, abs=0.02 * span_y)
    assert plot.box() == (t0, t1, y0, y1)


@pytest.mark.parametrize("button", [RIGHT, MIDDLE], ids=["right", "middle"])
def test_right_and_middle_drag_do_not_draw_a_box(plot, button):
    _prepare_for_drag(plot)
    spy = _Spy(plot.box_drawn)
    _drag(plot, _at(plot, T0 + 3.5 * HOUR, 35), _at(plot, T0 + 0.5 * HOUR, 5),
          button=button)
    assert spy.calls == []
    assert plot.box() is None


# ── outliers pinned to the edge ──────────────────────────────────────────────

def _bulk_with(outlier):
    return [float(v) for v in np.linspace(199, 201, 29)] + [outlier]


def test_high_outlier_is_one_pinned_marker_with_its_real_number(plot):
    plot.set_traces(_rows(_bulk_with(9957.0)))
    _settle(plot)
    lo, hi = _view_y(plot)
    assert hi < 300
    assert lo <= 199.0 and hi >= 201.0
    markers = _markers(plot)
    assert len(markers) == 1
    sc, marker = markers[0]
    assert lo <= marker.pos().y() <= hi
    assert marker.pos().y() >= hi - 0.1 * (hi - lo)       # at the top edge
    assert "9957" in _tip(sc, marker)
    assert len(_spots(plot)) == 30                         # one spot per trace


def test_low_outlier_is_pinned_to_the_bottom_edge(plot):
    plot.set_traces(_rows(_bulk_with(12.5)))
    _settle(plot)
    lo, hi = _view_y(plot)
    markers = _markers(plot)
    assert len(markers) == 1
    sc, marker = markers[0]
    assert lo <= marker.pos().y() <= lo + 0.1 * (hi - lo)
    assert "12.5" in _tip(sc, marker)


def test_no_marker_when_nothing_is_an_outlier(plot):
    plot.set_traces(_rows([float(v) for v in np.linspace(199, 201, 30)]))
    _settle(plot)
    assert not _markers(plot)


# ── guards on the display range ──────────────────────────────────────────────

def _assert_finite_view(plot):
    xr, yr = plot.getViewBox().viewRange()
    for lim in (*xr, *yr):
        assert math.isfinite(lim)
    assert xr[0] < xr[1] and yr[0] < yr[1]


def test_no_rows_gives_a_default_view_without_error(plot):
    plot.set_traces([])
    _settle(plot)
    assert _spots(plot) == []
    _assert_finite_view(plot)


def test_rows_all_without_own_value_give_a_finite_view(plot):
    plot.set_traces(_rows([None, None, None]))
    _settle(plot)
    assert _spots(plot) == []
    _assert_finite_view(plot)


def test_a_single_row_gives_a_finite_view_containing_it(plot):
    plot.set_traces(_rows([199.14]))
    _settle(plot)
    _assert_finite_view(plot)
    lo, hi = _view_y(plot)
    assert lo < 199.14 < hi
    assert len(_points(plot)) == 1


def test_non_finite_own_values_are_not_drawn_and_do_not_break_the_view(plot):
    plot.set_traces(_rows([200.0, float("nan"), 201.0, float("inf"), 199.0]))
    _settle(plot)
    _assert_finite_view(plot)
    for _, pt in _spots(plot):
        assert math.isfinite(pt.pos().x()) and math.isfinite(pt.pos().y())
    assert len(_points(plot)) == 3


def test_row_with_no_time_raises_nothing_and_adds_no_non_finite_point(plot):
    # The plan does not say whether a trace without an acquisition time is
    # drawn.  Only the safe part is asserted: no error, no point at a
    # non-finite position, the view stays finite, the timed rows are drawn.
    rows = [_row(0, 200.0), _row(1, 201.0, time_s=None), _row(2, 199.0)]
    plot.set_traces(rows)
    _settle(plot)
    for _, pt in _spots(plot):
        assert math.isfinite(pt.pos().x()) and math.isfinite(pt.pos().y())
    xs = {round(pt.pos().x()) for _, pt in _points(plot)}
    assert {round(T0), round(T0 + 2 * HOUR)} <= xs
    _assert_finite_view(plot)


# ── clicks report the index into the rows last given ─────────────────────────

def _click_at_spot(plot, spot, dy=0, double=False):
    p = _at(plot, spot.pos().x(), spot.pos().y())
    p = QPoint(p.x(), p.y() + dy)
    click = QTest.mouseDClick if double else QTest.mouseClick
    click(plot.viewport(), LEFT, pos=p)
    QApplication.processEvents()


def _spot_at_time(plot, time_s):
    for _, pt in _spots(plot):
        if math.isclose(pt.pos().x(), time_s, abs_tol=1.0):
            return pt
    raise AssertionError(f"no spot at time {time_s}")


def test_click_reports_the_row_index_even_when_an_earlier_row_is_undrawn(plot):
    rows = _rows([200.0, None, 201.0, 199.0])
    plot.set_traces(rows)
    _settle(plot)
    spy = _Spy(plot.point_selected)
    _click_at_spot(plot, _spot_at_time(plot, rows[2].time_s))
    assert spy.calls == [(2,)]


def test_double_click_opens_the_row_index_even_when_an_earlier_row_is_undrawn(plot):
    rows = _rows([200.0, None, 201.0, 199.0])
    plot.set_traces(rows)
    _settle(plot)
    opened = _Spy(plot.point_opened)
    _click_at_spot(plot, _spot_at_time(plot, rows[2].time_s), double=True)
    assert opened.calls == [(2,)]


def test_index_follows_the_rows_last_given(plot):
    plot.set_traces(_rows([200.0, 201.0, 199.0]))
    rows = [_row(5, 205.0), _row(6, None), _row(7, 206.0)]
    plot.set_traces(rows)
    _settle(plot)
    spy = _Spy(plot.point_selected)
    _click_at_spot(plot, _spot_at_time(plot, rows[2].time_s))
    assert spy.calls == [(2,)]


def test_click_on_a_pinned_marker_reports_its_row_index(plot):
    values = [None] + _bulk_with(9957.0)
    rows = _rows(values)
    plot.set_traces(rows)
    _settle(plot)
    (_, marker), = _markers(plot)
    spy = _Spy(plot.point_selected)
    _click_at_spot(plot, marker, dy=3)        # a little inside the edge
    assert spy.calls == [(len(rows) - 1,)]


def test_double_click_on_a_pinned_marker_opens_its_row_index(plot):
    rows = _rows([None] + _bulk_with(9957.0))
    plot.set_traces(rows)
    _settle(plot)
    (_, marker), = _markers(plot)
    opened = _Spy(plot.point_opened)
    _click_at_spot(plot, marker, dy=3, double=True)
    assert opened.calls == [(len(rows) - 1,)]


# ── a second set_traces leaves nothing behind ────────────────────────────────

def test_second_set_traces_leaves_no_leftovers(plot):
    first = _rows(_bulk_with(9957.0))
    first[3] = _row(3, first[3].own_nm_v, current=False)
    plot.set_traces(first)
    _settle(plot)
    assert _markers(plot) and len(_spots(plot)) == 30

    plot.set_traces(_rows([150.0, 151.0]))
    _settle(plot)
    assert len(_spots(plot)) == 2
    assert not _markers(plot)
    tips = " ".join(_tips(plot))
    assert "9957" not in tips
    assert "earlier fit settings" not in tips
    lo, hi = _view_y(plot)
    assert lo < 150.0 and hi > 151.0 and hi < 300


def test_set_traces_with_no_rows_after_rows_clears_everything(plot):
    plot.set_traces(_rows(_bulk_with(9957.0)))
    plot.set_traces([])
    _settle(plot)
    assert _spots(plot) == []


# ── the mean line ────────────────────────────────────────────────────────────

def test_mean_line_shows_for_a_value_and_goes_for_none(plot):
    plot.set_traces(_rows([199.0, 200.0, 201.0]))
    assert _horizontal_lines(plot, 200.0) == []
    plot.set_mean_line(200.0)
    assert len(_horizontal_lines(plot, 200.0)) == 1
    plot.set_mean_line(None)
    assert _horizontal_lines(plot, 200.0) == []


def test_mean_line_moves_to_a_new_value_and_never_doubles(plot):
    plot.set_traces(_rows([199.0, 200.0, 201.0]))
    plot.set_mean_line(200.0)
    plot.set_mean_line(200.5)
    assert _horizontal_lines(plot, 200.0) == []
    assert len(_horizontal_lines(plot, 200.5)) == 1


def test_mean_line_survives_a_new_set_traces_without_duplicating(plot):
    plot.set_mean_line(200.0)
    plot.set_traces(_rows([199.0, 200.0, 201.0]))
    assert len(_horizontal_lines(plot, 200.0)) <= 1
