# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
An upgrade that spans several versions is all or nothing.

If every step committed on its own, a catalog from before the version number
that failed in step 2 would stop at version 1, and the next open would back up
that half-upgraded copy over the only backup of the original.  So all pending
steps run in one transaction and the final version is stamped once.

The version-0 catalog is built by tests/catalog_v1.py, not by the code under
test.  Every call passes an explicit temporary path.
"""
import os
import sqlite3
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import catalog_v1                                            # noqa: E402
import invols_synth as synth                                 # noqa: E402
import run_dashboard as rd                                   # noqa: E402
from smfs_catalog import db                                  # noqa: E402


def _version0(tmp_path, name="v0.db"):
    path = str(tmp_path / name)
    catalog_v1.build_v0(path)
    con = sqlite3.connect(path)
    with con:
        con.execute("INSERT INTO files (path, filename, first_seen, last_seen, "
                    "inv_ols_nm_v, event) VALUES ('/v0/a.ibw', 'a.ibw', 'f', "
                    "'l', 199.14, 'event')")
        con.execute("INSERT INTO analysis_results VALUES "
                    "(1, 'invols_slope', -0.95, '{}', 'abc', 't')")
        con.execute("INSERT INTO meta VALUES ('built_host', 'somewhere')")
        con.execute("INSERT INTO meta VALUES "
                    "('criteria_bounds_reconciled_v1', 't')")
        con.execute("INSERT INTO meta VALUES "
                    "('criteria_integer_bounds_whole_v1', 't')")
    con.close()
    return path


def _columns(path):
    """Every table and its sorted columns."""
    con = sqlite3.connect(path)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: sorted(r[1] for r in con.execute(f"PRAGMA table_info({t})"))
                for t in tables}
    finally:
        con.close()


def _rows(path):
    con = sqlite3.connect(path)
    try:
        return {t: sorted(con.execute(f"SELECT * FROM {t}").fetchall(), key=repr)
                for t in _columns(path)}
    finally:
        con.close()


def _version(path):
    return synth.fetch(path, "PRAGMA user_version")[0][0]


def _fails_in_step_2(monkeypatch):
    def fails(conn):
        raise RuntimeError("interrupted in step 2")
    monkeypatch.setitem(db._UPGRADES, 2, fails)


# ── a. A failure in step 2 undoes step 1 as well ──────────────────────────────

def test_a_failure_in_step_2_leaves_a_version_0_catalog_untouched(
        tmp_path, monkeypatch):
    path = _version0(tmp_path)
    columns, rows = _columns(path), _rows(path)
    _fails_in_step_2(monkeypatch)
    with pytest.raises(RuntimeError):
        db.initialise(path)
    assert _version(path) == 0
    assert _columns(path) == columns          # step 1's changes were undone
    assert _rows(path) == rows


# ── b. The retry through open_catalog keeps the backup of the original ────────

class _Dialogs:
    def __init__(self):
        self.refused = []

    def confirm(self, text):
        return True

    def refuse(self, text):
        self.refused.append(text)


def test_a_retried_upgrade_keeps_the_backup_of_the_original(tmp_path, monkeypatch):
    path = _version0(tmp_path)
    columns, rows = _columns(path), _rows(path)
    dialogs = _Dialogs()

    with monkeypatch.context() as broken:
        _fails_in_step_2(broken)
        with pytest.raises(RuntimeError):
            rd.open_catalog(path, dialogs.confirm, dialogs.refuse)

    assert rd.open_catalog(path, dialogs.confirm, dialogs.refuse) is True
    assert _version(path) == db.CATALOG_VERSION

    backup = db.backup_path(path)
    assert _version(backup) == 0
    assert _columns(backup) == columns
    assert _rows(backup) == rows
    assert dialogs.refused == []


# ── c. The paths that work still work ─────────────────────────────────────────

def test_a_version_0_catalog_reaches_version_2(tmp_path):
    path = _version0(tmp_path)
    fresh_path = str(tmp_path / "fresh.db")
    db.initialise(fresh_path)

    db.initialise(path)

    assert _version(path) == 2
    assert _columns(path) == _columns(fresh_path)
    assert synth.fetch(path, "SELECT path, inv_ols_nm_v FROM files") == [
        ("/v0/a.ibw", 199.14)]
    transform = synth.fetch(
        path, "SELECT value FROM analysis_results WHERE file_id = 1 "
        "AND analysis_type = 'transform_invols_nm_v'")
    assert transform[0][0] == pytest.approx(199.14 / 0.95, rel=1e-9)


def test_a_version_1_catalog_still_upgrades_to_version_2(tmp_path):
    path = str(tmp_path / "v1.db")
    catalog_v1.build_v1(path)
    db.initialise(path)
    assert _version(path) == 2
    assert {"invols_groups", "invols_group_members",
            "invols_assignments"} <= set(_columns(path))


def test_a_catalog_that_fails_in_step_2_can_be_upgraded_afterwards(
        tmp_path, monkeypatch):
    path = _version0(tmp_path)
    with monkeypatch.context() as broken:
        _fails_in_step_2(broken)
        with pytest.raises(RuntimeError):
            db.initialise(path)
    db.initialise(path)
    assert _version(path) == 2
