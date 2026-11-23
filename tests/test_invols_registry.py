# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Registry entry for Transform InvOLS: the nm/V value a trace's deflection is
converted with (its own per-curve fit, or an ensemble mean applied from the
InvOLS window).  Name, unit, description and position come from the one
registry (variables.py, quantities.py); nothing else may keep a copy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smfs_catalog import quantities as _q               # noqa: E402
from smfs_catalog import variables as _vars             # noqa: E402

KEY = "transform_invols_nm_v"


def test_label_is_transform_invols():
    assert _vars.label(KEY) == "Transform InvOLS"


def test_labelled_carries_the_unit():
    assert _vars.labelled(KEY) == "Transform InvOLS (nm/V)"


def test_unit_is_nm_per_volt():
    assert _q.unit_of(KEY) == "nm/V"


def test_source_is_analysis_result():
    assert _vars.source_of(KEY) == _vars.SOURCE_ANALYSIS


def test_listed_in_display_order():
    assert KEY in _vars.DISPLAY_ORDER


def test_sits_directly_after_invols_rms():
    order = _vars.DISPLAY_ORDER
    assert KEY in order, "not in DISPLAY_ORDER"
    assert order.index(KEY) == order.index("invols_rms") + 1


def test_format_value_two_decimals():
    assert _q.format_value(KEY, 199.14159) == "199.14"


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_format_value_missing_is_blank(missing):
    assert _q.format_value(KEY, missing) == ""


def test_description_is_present():
    assert _vars.describe(KEY).strip() != ""


@pytest.mark.parametrize("word", ["accurate", "calibrated", "true"])
def test_description_avoids_claiming_words(word):
    text = _vars.describe(KEY)
    assert text, "no description to check"
    assert word not in text.lower()
