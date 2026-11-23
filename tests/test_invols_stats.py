"""Tests for smfs_catalog.invols_stats: pure invOLS arithmetic.

Expected values come from hand calculation or the plan's worked examples.
"""
import random

import numpy as np
import pytest

from smfs_catalog.invols_stats import (
    AVERAGES,
    DIVISOR_SIGN,
    applied_value,
    box_cut_count,
    cache_key,
    display_range,
    ensemble_divisor,
    own_transform_nm_v,
    points_in_box,
    resolve_divisor,
    summarise,
)


# ---------------------------------------------------------------- constants

def test_constants():
    assert DIVISOR_SIGN == -1.0
    assert AVERAGES == ("arithmetic", "harmonic")


# ------------------------------------------------------ own_transform_nm_v

def test_own_transform_divides_instrument_by_abs_slope():
    assert own_transform_nm_v(199.14, -0.9482857142857143) == pytest.approx(210.0, rel=1e-12)


def test_own_transform_unit_slope_returns_instrument_value():
    assert own_transform_nm_v(199.14, -1.0) == pytest.approx(199.14, rel=1e-15)


def test_own_transform_shallow_slope_is_not_hidden():
    assert own_transform_nm_v(199.14, -0.02) == pytest.approx(9957.0, rel=1e-12)


@pytest.mark.parametrize("instrument, slope", [
    (199.14, 0.9),            # wrong sign
    (199.14, 0.0),
    (199.14, float("nan")),
    (199.14, float("inf")),
    (199.14, None),
    (None, -0.95),
    (0.0, -0.95),
    (-5.0, -0.95),
    (float("nan"), -0.95),
    (199.14, -5e-324),        # subnormal slope: quotient is infinite
])
def test_own_transform_none_cases(instrument, slope):
    assert own_transform_nm_v(instrument, slope) is None


# ---------------------------------------------------------------- summarise

def test_summarise_research_r3_example():
    values = [own_transform_nm_v(199.14, s) for s in (-0.95, -1.00, -1.05)]
    assert values[0] == pytest.approx(209.62105, rel=1e-6)
    assert values[1] == pytest.approx(199.14, rel=1e-12)
    assert values[2] == pytest.approx(189.65714, rel=1e-6)
    s = summarise(values)
    assert s.n == 3
    assert s.mean == pytest.approx(199.47273, rel=1e-6)
    assert s.sd == pytest.approx(9.98611, rel=1e-5)
    assert s.se == pytest.approx(5.7655, rel=1e-4)
    assert s.harmonic_mean == pytest.approx(199.14, rel=1e-12)


def test_summarise_three_values():
    s = summarise([50, 52, 54])
    assert s.n == 3
    assert s.mean == pytest.approx(52.0, rel=1e-15)
    assert s.sd == pytest.approx(2.0, rel=1e-12)
    assert s.se == pytest.approx(1.1547005383792517, rel=1e-12)
    assert s.harmonic_mean == pytest.approx(51.9487, rel=1e-4)


def test_summarise_harmonic_of_100_and_200():
    s = summarise([100, 200])
    assert s.mean == pytest.approx(150.0, rel=1e-15)
    assert s.harmonic_mean == pytest.approx(400 / 3, rel=1e-12)


def test_summarise_two_values_sd_and_se():
    s = summarise([50, 54])
    assert s.sd == pytest.approx(2.8284271247461903, rel=1e-12)
    assert s.se == pytest.approx(2.0, rel=1e-12)


def test_summarise_single_value_has_no_sd_or_se():
    s = summarise([50])
    assert s.n == 1
    assert s.mean == pytest.approx(50.0, rel=1e-15)
    assert s.harmonic_mean == pytest.approx(50.0, rel=1e-15)
    assert s.sd is None
    assert s.se is None


def test_summarise_empty_is_all_none():
    s = summarise([])
    assert s.n == 0
    assert s.mean is None
    assert s.harmonic_mean is None
    assert s.sd is None
    assert s.se is None


def test_summarise_drops_non_finite_and_none():
    s = summarise([50, float("nan"), 54, None, float("inf")])
    assert s.n == 2
    assert s.mean == pytest.approx(52.0, rel=1e-15)


def test_summarise_non_positive_value_blocks_harmonic_only():
    s = summarise([50, -3])
    assert s.harmonic_mean is None
    assert s.mean == pytest.approx(23.5, rel=1e-15)


def test_summarise_zero_value_blocks_harmonic():
    assert summarise([50, 0]).harmonic_mean is None


def test_summarise_identical_values_sd_near_zero():
    s = summarise([0.1] * 3)
    assert s.sd == pytest.approx(0.0, abs=1e-12)


def test_summarise_is_order_independent():
    rng = random.Random(12345)
    values = [rng.uniform(150.0, 250.0) * (1 + 1e-9 * i) for i in range(137)]
    ref = summarise(values)
    for _ in range(50):
        shuffled = values[:]
        rng.shuffle(shuffled)
        s = summarise(shuffled)
        assert s.mean == ref.mean
        assert s.harmonic_mean == ref.harmonic_mean


def test_summarise_outlier_mean_hand_calculation():
    s = summarise([200.0] * 29 + [9800.0])
    assert s.n == 30
    assert s.mean == pytest.approx(520.0, rel=1e-12)


# ------------------------------------------------------------ applied_value

def test_applied_value_arithmetic():
    assert applied_value(summarise([100, 200]), "arithmetic") == pytest.approx(150.0, rel=1e-15)


def test_applied_value_harmonic():
    assert applied_value(summarise([100, 200]), "harmonic") == pytest.approx(400 / 3, rel=1e-12)


def test_applied_value_unknown_average_raises():
    with pytest.raises(ValueError):
        applied_value(summarise([100, 200]), "median")


# --------------------------------------------------------- ensemble_divisor

def test_ensemble_divisor_worked_example():
    assert ensemble_divisor(199.14, 210.0) == pytest.approx(-0.9482857142857143, rel=1e-15)


def test_ensemble_divisor_rounding_cancels():
    ratio = 199.1 / 199.14
    a = ensemble_divisor(199.14, 210.0)
    b = ensemble_divisor(199.1, 210.0 * ratio)
    assert b == pytest.approx(a, rel=1e-15)


# ---------------------------------------------------------- resolve_divisor

def test_resolve_own_slope_exact():
    assert resolve_divisor(-0.95, None) == -0.95


def test_resolve_ensemble_wins():
    assert resolve_divisor(-0.95, -0.9483) == -0.9483


@pytest.mark.parametrize("slope", [float("nan"), None, 0.0])
def test_resolve_no_usable_slope_no_ensemble_is_none(slope):
    assert resolve_divisor(slope, None) is None


def test_resolve_ensemble_used_when_own_slope_unusable():
    assert resolve_divisor(float("nan"), -0.9483) == -0.9483


def test_resolve_positive_slope_kept():
    assert resolve_divisor(0.9, None) == 0.9


# ---------------------------------------------------------------- cache_key

def test_cache_key_without_ensemble_is_unchanged():
    assert cache_key("P", None) == "P"


def test_cache_key_with_ensemble():
    assert cache_key("P", -0.95) == "P|invols:-0.95"


def test_cache_key_same_float_same_text():
    assert cache_key("P", -0.9482857142857143) == cache_key("P", -0.9482857142857143)


def test_cache_key_differs_for_different_divisors():
    assert cache_key("P", -0.95) != cache_key("P", -0.96)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_cache_key_non_finite_raises(bad):
    with pytest.raises(ValueError):
        cache_key("P", bad)


# ------------------------------------------------------------ display_range

def test_display_range_flags_the_outlier():
    values = np.linspace(199, 201, 29).tolist() + [9957]
    r = display_range(values)
    assert len(r.side) == 30
    assert r.side[-1] == 1
    assert all(x == 0 for x in r.side[:-1])
    assert 198.5 <= r.lo <= 199.0
    assert 201.0 <= r.hi <= 201.5


def test_display_range_flags_a_low_outlier():
    values = [-5000.0] + np.linspace(199, 201, 29).tolist()
    r = display_range(values)
    assert r.side[0] == -1
    assert all(x == 0 for x in r.side[1:])


def test_display_range_zero_span_pads_one_percent():
    r = display_range([200, 200, 200])
    assert (r.lo, r.hi) == (198.0, 202.0)
    assert r.side == (0, 0, 0)


def test_display_range_single_zero():
    r = display_range([0.0])
    assert (r.lo, r.hi) == (-1.0, 1.0)


# ------------------------------------------------------------ box helpers

T = [0, 1, 2, 3]
Y = [10, 20, 30, 40]


def test_points_in_box():
    mask = points_in_box(T, Y, (1, 2, 15, 35))
    assert list(mask) == [False, True, True, False]


def test_points_in_box_reversed_corners_same_mask():
    assert list(points_in_box(T, Y, (2, 1, 35, 15))) == [False, True, True, False]


def test_points_in_box_edges_inclusive():
    assert list(points_in_box(T, Y, (1, 2, 20, 30))) == [False, True, True, False]


def test_box_cut_count():
    assert box_cut_count(T, Y, (0, 3, 15, 35)) == 2


def test_box_cut_count_zero_when_nothing_cut():
    assert box_cut_count(T, Y, (0, 3, 5, 45)) == 0


def test_box_cut_count_ignores_points_outside_the_time_span():
    # times 1 and 2 are in the span: 20 is below the limit, 30 is inside.
    # times 0 and 3 are out of span, so they are not counted whatever their value.
    assert box_cut_count(T, Y, (1, 2, 25, 35)) == 1


def test_box_cut_count_does_not_count_a_missing_value_none():
    assert box_cut_count([0, 1, 2], [20.0, None, 20.0], (0, 2, 15, 35)) == 0


def test_box_cut_count_does_not_count_a_missing_value_nan():
    assert box_cut_count([0, 1, 2], [20.0, float("nan"), 20.0], (0, 2, 15, 35)) == 0


def test_display_range_pads_five_percent_of_the_span():
    r = display_range([10.0, 20.0])
    assert r.lo == pytest.approx(9.5, rel=1e-12)
    assert r.hi == pytest.approx(20.5, rel=1e-12)


# ------------------------------------------------------ usable_instrument

def test_a_finite_positive_instrument_value_is_usable():
    from smfs_catalog import invols_stats
    assert invols_stats.usable_instrument(199.14) is True


@pytest.mark.parametrize("value", [
    None, 0, 0.0, -100.0, float("nan"), float("inf"), float("-inf")])
def test_no_value_zero_negative_and_non_finite_are_not_usable(value):
    from smfs_catalog import invols_stats
    assert invols_stats.usable_instrument(value) is False
