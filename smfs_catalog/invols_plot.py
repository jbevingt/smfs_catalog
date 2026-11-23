"""The invOLS plot: one point per trace, time on x and nm/V on y, with a box
drawn by a left drag.  It takes ``TraceRow``s and touches no database."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal

from . import quantities, style, variables
from .invols_stats import TRANSFORM_KEY, display_range
from .qt_utils import _DateAxis, set_si_label

if TYPE_CHECKING:
    from .invols_store import TraceRow

_DEFAULT_Y = (0.0, 1.0)
_DEFAULT_X_HALF_SPAN_S = 3600.0
_SINGLE_TIME_HALF_SPAN_S = 60.0
_PIN_INSET = 0.02
_POINT_SIZE = 8
_MARKER_SIZE = 11
_SOURCE_TEXT = {"own": "own fit", "ensemble": "ensemble value"}


class _BoxViewBox(pg.ViewBox):
    box_dragged = pyqtSignal(float, float, float, float, bool)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        start = self.mapToView(ev.buttonDownPos())
        now = self.mapToView(ev.pos())
        self.box_dragged.emit(start.x(), now.x(), start.y(), now.y(), ev.isFinish())


def _drawn(rows: Sequence[TraceRow]) -> list[tuple[int, TraceRow]]:
    return [(i, r) for i, r in enumerate(rows)
            if r.own_nm_v is not None and math.isfinite(r.own_nm_v)
            and r.time_s is not None and math.isfinite(r.time_s)]


def _y_view(values: list[float]) -> tuple[float, float, tuple[int, ...]]:
    if not values:
        return (*_DEFAULT_Y, ())
    shown = display_range(values)
    if not (math.isfinite(shown.lo) and math.isfinite(shown.hi) and shown.hi > shown.lo):
        return (*_DEFAULT_Y, (0,) * len(values))
    return shown.lo, shown.hi, shown.side


def _x_view(times: list[float]) -> tuple[float, float]:
    if not times:
        now = time.time()
        return now - _DEFAULT_X_HALF_SPAN_S, now + _DEFAULT_X_HALF_SPAN_S
    lo, hi = min(times), max(times)
    pad = 0.05 * (hi - lo) or _SINGLE_TIME_HALF_SPAN_S
    return lo - pad, hi + pad


def _tip(x, y, data) -> str:
    unit = quantities.unit_of(TRANSFORM_KEY)
    lines = [data["name"],
             f"{variables.label(TRANSFORM_KEY)}: {data['value']:.2f} {unit} "
             f"({_SOURCE_TEXT.get(data['source'], data['source'])})"]
    if data["side"]:
        lines.append("Outside the plotted range, shown at its edge")
    if not data["current"]:
        lines.append("Fit made under earlier fit settings")
    return "\n".join(lines)


def _spot(index: int, row: TraceRow, side: int, lo: float, hi: float) -> dict:
    colour = style.INVOLS_ENSEMBLE if row.source == "ensemble" else style.INVOLS_OWN
    inset = _PIN_INSET * (hi - lo)
    y = hi - inset if side > 0 else lo + inset if side < 0 else row.own_nm_v
    return {
        "pos": (row.time_s, y),
        "symbol": "t1" if side > 0 else "t" if side < 0 else "o",
        "size": _MARKER_SIZE if side else _POINT_SIZE,
        "pen": pg.mkPen(colour, width=1.5),
        "brush": pg.mkBrush(colour) if row.fit_current else pg.mkBrush(None),
        "data": {"index": index, "value": row.own_nm_v, "side": side,
                 "current": row.fit_current, "source": row.source,
                 "name": Path(row.path).name},
    }


class InvolsPlot(pg.PlotWidget):
    box_drawn = pyqtSignal(float, float, float, float)
    point_selected = pyqtSignal(int)
    point_opened = pyqtSignal(int)

    def __init__(self, parent=None):
        self._view_box = _BoxViewBox()
        super().__init__(parent, viewBox=self._view_box,
                         axisItems={"bottom": _DateAxis()})
        set_si_label(self, "left", variables.label(TRANSFORM_KEY),
                     quantities.unit_of(TRANSFORM_KEY), si=False)
        self._view_box.disableAutoRange()
        self._scatter = None
        self._mean_line = None
        self._box = None
        self._box_item = pg.QtWidgets.QGraphicsRectItem()
        self._box_item.setPen(pg.mkPen(style.INK_STRONG, width=1))
        self._box_item.setVisible(False)
        self._view_box.addItem(self._box_item, ignoreBounds=True)
        self._view_box.box_dragged.connect(self._on_box_dragged)
        self._apply_view([], [])

    def set_traces(self, rows: Sequence[TraceRow]) -> None:
        if self._scatter is not None:
            self._view_box.removeItem(self._scatter)
            self._scatter = None
        self._view_box.setToolTip("")
        drawn = _drawn(rows)
        lo, hi, side = _y_view([r.own_nm_v for _, r in drawn])
        self._apply_view([r.time_s for _, r in drawn], [lo, hi])
        if not drawn:
            return
        spots = [_spot(i, r, s, lo, hi) for (i, r), s in zip(drawn, side)]
        self._scatter = pg.ScatterPlotItem(spots=spots, tip=_tip, hoverable=True)
        self._scatter.sigClicked.connect(self._on_clicked)
        self._view_box.addItem(self._scatter)

    def _apply_view(self, times: list[float], y_limits: list[float]) -> None:
        x0, x1 = _x_view(times)
        y0, y1 = (y_limits[0], y_limits[1]) if y_limits else _DEFAULT_Y
        self._view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def _on_clicked(self, _scatter, points, ev) -> None:
        if not ev.double():
            self.point_selected.emit(int(points[0].data()["index"]))

    def mouseDoubleClickEvent(self, ev) -> None:
        super().mouseDoubleClickEvent(ev)
        if self._scatter is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        local = self._scatter.mapFromScene(self.mapToScene(ev.position().toPoint()))
        points = self._scatter.pointsAt(local)
        if len(points):
            self.point_opened.emit(int(points[0].data()["index"]))

    def _on_box_dragged(self, t0, t1, y0, y1, finished) -> None:
        if finished:
            self.commit_box(t0, t1, y0, y1)
        else:
            self._show_box(t0, t1, y0, y1)

    def _show_box(self, t0, t1, y0, y1) -> None:
        self._box_item.setRect(min(t0, t1), min(y0, y1), abs(t1 - t0), abs(y1 - y0))
        self._box_item.setVisible(True)

    def commit_box(self, t0: float, t1: float, y0: float, y1: float) -> None:
        self._box = (float(min(t0, t1)), float(max(t0, t1)),
                     float(min(y0, y1)), float(max(y0, y1)))
        self._show_box(*self._box)
        self.box_drawn.emit(*self._box)

    def box(self) -> tuple[float, float, float, float] | None:
        return self._box

    def clear_box(self) -> None:
        self._box = None
        self._box_item.setVisible(False)

    def set_mean_line(self, value: float | None) -> None:
        if self._mean_line is not None:
            self._view_box.removeItem(self._mean_line)
            self._mean_line = None
        if value is None:
            return
        pen = pg.mkPen(style.INK_STRONG, width=1, style=Qt.PenStyle.DashLine)
        self._mean_line = pg.InfiniteLine(pos=value, angle=0, movable=False, pen=pen)
        self._view_box.addItem(self._mean_line)
