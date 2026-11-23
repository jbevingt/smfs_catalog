# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Catalog version 2 and its database helpers: the three invOLS tables, the
upgrade that backfills each file's own Transform InvOLS, the divisor an applied
ensemble gives a file, the per-file cache key, and the freshness it drives.

Expected values come from the plan's hand calculations (for instance
199.14 / 0.95 = 209.6210526...), never from running the code.  Every call
passes an explicit temporary catalog.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import catalog_v1                                            # noqa: E402
import invols_synth as synth                                 # noqa: E402
from smfs_catalog import db                                  # noqa: E402

INSTRUMENT = 199.14
GROUP_VALUE = 210.0
NEW_TABLES = ("invols_groups", "invols_group_members", "invols_assignments")


def _catalog(tmp_path, n=2):
    """`n` queued files with the same instrument value; file 1 is assigned to a
    group whose value is 210, the others are on their own fit."""
    files = [{"instrument": INSTRUMENT, "slope": -0.95, "event_params": "P"}
             for _ in range(n)]
    db_path, ids = synth.make_catalog(tmp_path, files)
    synth.add_group(db_path, [ids[0]], GROUP_VALUE)
    return db_path, ids


# ── The version and the upgrade ────────────────────────────────────────────────

def test_the_catalog_version_is_2():
    assert db.CATALOG_VERSION == 2


def test_a_new_catalog_has_the_three_tables(tmp_path):
    db_path, _ = synth.make_catalog(tmp_path, [])
    found = {r[0] for r in synth.fetch(
        db_path, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert set(NEW_TABLES) <= found


V1_SLOPE_PARAMS = '{"invols_offset_pts": 7, "invols_window_pts": 90}'
V1_SLOPE_CODE = "abc123"

# (path, instrument, slope): A, B, C, D and E of the plan.
V1_FILES = (
    ("/v1/A.ibw", 199.14, -0.95),
    ("/v1/B.ibw", 199.14, 0.9),
    ("/v1/C.ibw", 199.14, 0.0),
    ("/v1/D.ibw", None, -0.95),
    ("/v1/E.ibw", 199.14, None),
)
# Tables an upgrade is allowed to add rows to on its own.
V1_OPEN_TABLES = ("meta", "thresholds", "experimentalist_profiles",
                  "analysis_results")


def _fill_version_1(path):
    """Rows in every table a catalog of that age could hold, written by hand."""
    con = sqlite3.connect(path)
    with con:
        for number, (name, instrument, slope) in enumerate(V1_FILES, start=1):
            con.execute(
                "INSERT INTO files (id, path, filename, first_seen, last_seen, "
                "inv_ols_nm_v, event, measured_at) VALUES (?, ?, ?, 'f', 'l', "
                "?, 'event', ?)",
                (number, name, name[4:], instrument, f"2026-01-0{number}"))
            con.execute("INSERT INTO analysis_results VALUES "
                        "(?, 'invols_slope', ?, ?, ?, 't1')",
                        (number, slope, V1_SLOPE_PARAMS, V1_SLOPE_CODE))
            con.execute("INSERT INTO analysis_results VALUES "
                        "(?, 'event', 1.0, 'P', 'zzz', 't2')", (number,))
            con.execute("INSERT INTO analysis_queue VALUES (?, 'q', 'pending')",
                        (number,))
            con.execute("INSERT INTO file_metadata VALUES (?, 'k', 2.5, 'v')",
                        (number,))
            con.execute("INSERT INTO event_map VALUES (?, 'P', 'zzz', '{}', 't3')",
                        (number,))
        con.execute("INSERT INTO meta VALUES ('built_host', 'somewhere')")
        con.execute("INSERT INTO settings VALUES ('s', 1.5, 't')")
        con.execute("INSERT INTO app_settings VALUES ('a', 'b', 't')")
        con.execute("INSERT INTO distribution_fits (variable, n_values, n_peaks, "
                    "model_label, params_json, gof_json, created_at) VALUES "
                    "('v', 3, 1, 'm', '{}', '{}', 't')")
    con.close()


def _snapshot(path):
    """Every row of every table, as sorted lists of tuples."""
    con = sqlite3.connect(path)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'")]
        return {t: sorted(con.execute(f"SELECT * FROM {t}").fetchall(), key=repr)
                for t in tables}
    finally:
        con.close()


@pytest.fixture
def upgraded(tmp_path):
    path = str(tmp_path / "v1.db")
    catalog_v1.build_v1(path)
    _fill_version_1(path)
    before = _snapshot(path)
    db.initialise(path)
    return path, before, _snapshot(path)


def test_a_version_1_catalog_is_brought_to_version_2(upgraded):
    path, _, _ = upgraded
    assert synth.fetch(path, "PRAGMA user_version") == [(2,)]


def test_the_upgrade_gives_file_a_its_own_transform_invols(upgraded):
    path, _, _ = upgraded
    rows = synth.fetch(
        path, "SELECT value, params_json, code_version, computed_at "
        "FROM analysis_results WHERE file_id = 1 "
        "AND analysis_type = 'transform_invols_nm_v'")
    assert len(rows) == 1
    value, params, code, computed_at = rows[0]
    assert value == pytest.approx(209.6210526, rel=1e-9)      # 199.14 / 0.95
    assert params == V1_SLOPE_PARAMS
    assert code == V1_SLOPE_CODE
    assert computed_at


@pytest.mark.parametrize("file_id, why", [
    (2, "slope +0.9 has the wrong sign"), (3, "slope 0.0"),
    (4, "no instrument value"), (5, "slope NULL")])
def test_the_upgrade_gives_no_value_where_there_is_none(upgraded, file_id, why):
    path, _, _ = upgraded
    assert synth.fetch(
        path, "SELECT * FROM analysis_results WHERE file_id = ? "
        "AND analysis_type = 'transform_invols_nm_v'", (file_id,)) == [], why


def test_the_upgrade_adds_nothing_but_the_one_row(upgraded):
    _, _, after = upgraded
    added = [r for r in after["analysis_results"]
             if r[1] == "transform_invols_nm_v"]
    assert len(added) == 1
    assert len(after["analysis_results"]) == 10 + 1       # 5 slope + 5 event + 1


def test_every_pre_existing_row_is_unchanged(upgraded):
    _, before, after = upgraded
    for table, rows in before.items():
        if table in V1_OPEN_TABLES:
            continue
        assert after[table] == rows, table
    for table in ("meta", "analysis_results"):
        assert all(row in after[table] for row in before[table]), table


def test_the_upgrade_leaves_the_new_tables_empty(upgraded):
    _, _, after = upgraded
    assert all(after[t] == [] for t in NEW_TABLES)


def test_a_catalog_two_versions_ahead_is_refused_and_left_unchanged(tmp_path):
    db_path, _ = synth.make_catalog(tmp_path, [])
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA user_version = 3")
    con.close()
    before = Path(db_path).read_bytes()
    with pytest.raises(db.CatalogTooNew):
        db.initialise(db_path)
    assert Path(db_path).read_bytes() == before


# ── ensemble_divisors, divisor_for_file, file_params_key_for ──────────────────

def test_ensemble_divisors_gives_the_assigned_file_its_divisor(tmp_path):
    db_path, ids = _catalog(tmp_path)
    got = db.ensemble_divisors(ids, db_path=db_path)
    assert list(got) == [ids[0]]
    assert got[ids[0]] == pytest.approx(-199.14 / 210, rel=1e-15)


def test_ensemble_divisors_of_none_returns_every_assigned_file(tmp_path):
    files = [{"instrument": 199.14}, {"instrument": 150.0},
             {"instrument": 180.0}]
    db_path, ids = synth.make_catalog(tmp_path, files)
    synth.add_group(db_path, [ids[0]], 210.0)
    synth.add_group(db_path, [ids[2]], 190.0)
    got = db.ensemble_divisors(None, db_path=db_path)
    assert set(got) == {ids[0], ids[2]}
    assert got[ids[0]] == pytest.approx(-199.14 / 210.0, rel=1e-15)
    assert got[ids[2]] == pytest.approx(-180.0 / 190.0, rel=1e-15)


def test_ensemble_divisors_uses_a_connection_it_is_given(tmp_path):
    db_path, ids = _catalog(tmp_path)
    conn = db.get_connection(db_path)
    try:
        got = db.ensemble_divisors(ids, db_path=db_path, conn=conn)
    finally:
        conn.close()
    assert got[ids[0]] == pytest.approx(-199.14 / 210, rel=1e-15)


def test_divisor_for_file_is_the_ensemble_divisor_when_assigned(tmp_path):
    db_path, ids = _catalog(tmp_path)
    got = db.divisor_for_file(ids[0], -0.95, db_path=db_path)
    assert got == pytest.approx(-199.14 / 210, rel=1e-15)


def test_divisor_for_file_is_the_own_slope_when_not_assigned(tmp_path):
    db_path, ids = _catalog(tmp_path)
    assert db.divisor_for_file(ids[1], -0.95, db_path=db_path) == -0.95


def test_divisor_for_file_is_none_without_an_own_slope_or_group(tmp_path):
    db_path, ids = _catalog(tmp_path)
    assert db.divisor_for_file(ids[1], None, db_path=db_path) is None


def test_file_params_key_is_unchanged_for_an_own_fit_file(tmp_path):
    db_path, ids = _catalog(tmp_path)
    assert db.file_params_key_for(ids[1], "P", db_path=db_path) == "P"


def test_file_params_key_carries_the_divisor_for_an_ensemble_file(tmp_path):
    db_path, ids = _catalog(tmp_path)
    expected = "P|invols:" + json.dumps(-199.14 / 210)
    assert db.file_params_key_for(ids[0], "P", db_path=db_path) == expected


# ── queue_freshness ────────────────────────────────────────────────────────────

def test_an_ensemble_file_is_stale_until_rewritten_under_its_own_key(tmp_path):
    db_path, ids = _catalog(tmp_path)
    assert db.queue_freshness("P", "C", db_path) == {
        ids[0]: "stale", ids[1]: "fresh"}

    key = db.file_params_key_for(ids[0], "P", db_path=db_path)
    db.write_analysis_result(ids[0], "event", 1.0, key, "C", db_path)
    assert db.queue_freshness("P", "C", db_path) == {
        ids[0]: "fresh", ids[1]: "fresh"}


def test_a_file_with_no_event_row_is_new(tmp_path):
    db_path, ids = synth.make_catalog(
        tmp_path, [{"instrument": INSTRUMENT, "event_params": "P"},
                   {"instrument": INSTRUMENT}])
    assert db.queue_freshness("P", "C", db_path) == {
        ids[0]: "fresh", ids[1]: "new"}


def test_a_changed_code_version_is_still_stale_for_an_own_fit_file(tmp_path):
    db_path, ids = _catalog(tmp_path)
    assert db.queue_freshness("P", "OTHER", db_path)[ids[1]] == "stale"


# ── write_analysis_results_multi(commit=False) ────────────────────────────────

def _written_rows(db_path, file_id):
    return synth.fetch(
        db_path, "SELECT value FROM analysis_results WHERE file_id = ? "
        "AND analysis_type = 'transform_invols_nm_v'", (file_id,))


def test_commit_false_inside_a_rolled_back_transaction_leaves_no_row(tmp_path):
    db_path, ids = synth.make_catalog(tmp_path, [{"instrument": INSTRUMENT}])
    conn = db.get_connection(db_path)
    try:
        db.write_analysis_results_multi(
            ids[0], {"transform_invols_nm_v": 210.0}, "P", "C",
            db_path=db_path, conn=conn, commit=False)
        conn.rollback()
    finally:
        conn.close()
    assert _written_rows(db_path, ids[0]) == []


def test_commit_false_writes_nothing_durable_until_the_caller_commits(tmp_path):
    db_path, ids = synth.make_catalog(tmp_path, [{"instrument": INSTRUMENT}])
    conn = db.get_connection(db_path)
    try:
        db.write_analysis_results_multi(
            ids[0], {"transform_invols_nm_v": 210.0}, "P", "C",
            db_path=db_path, conn=conn, commit=False)
        assert _written_rows(db_path, ids[0]) == []        # seen from outside
        conn.commit()
    finally:
        conn.close()
    assert _written_rows(db_path, ids[0]) == [(210.0,)]


def test_the_default_still_commits_on_a_given_connection(tmp_path):
    db_path, ids = synth.make_catalog(tmp_path, [{"instrument": INSTRUMENT}])
    conn = db.get_connection(db_path)
    try:
        db.write_analysis_results_multi(
            ids[0], {"transform_invols_nm_v": 210.0}, "P", "C",
            db_path=db_path, conn=conn)
        assert _written_rows(db_path, ids[0]) == [(210.0,)]
    finally:
        conn.close()


# ── Erasing analysis, removing files ──────────────────────────────────────────

def _table_file_ids(db_path, table):
    return {r[0] for r in synth.fetch(db_path, f"SELECT file_id FROM {table}")}


def _two_assigned(tmp_path):
    files = [{"instrument": INSTRUMENT, "slope": -0.95} for _ in range(2)]
    db_path, ids = synth.make_catalog(tmp_path, files)
    synth.add_group(db_path, ids, GROUP_VALUE)
    return db_path, ids


def test_erasing_a_files_analysis_deletes_its_assignment(tmp_path):
    db_path, ids = _two_assigned(tmp_path)
    db.erase_analysis_for_files([synth.path_of(1)], db_path)
    assert _table_file_ids(db_path, "invols_assignments") == {ids[1]}
    assert _table_file_ids(db_path, "invols_group_members") == set(ids)
    assert _table_file_ids(db_path, "analysis_results") == {ids[1]}


def test_removing_a_file_deletes_its_assignment_and_member_rows(tmp_path):
    db_path, ids = _two_assigned(tmp_path)
    db.remove_files_from_catalog([synth.path_of(1)], db_path)
    assert _table_file_ids(db_path, "invols_assignments") == {ids[1]}
    assert _table_file_ids(db_path, "invols_group_members") == {ids[1]}
    assert synth.fetch(db_path, "SELECT id FROM files") == [(ids[1],)]
