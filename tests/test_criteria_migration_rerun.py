"""Precondition for D8: reconciling an already-reconciled owner changes nothing.

D8 makes `reconcile` skip its done-marker when one owner's profile is corrupt,
so the next launch runs the whole reconcile again, including over owners that
were reconciled on the first pass.  That is only safe if a second pass over an
already-reconciled owner is a no-op.  The marker normally hides the question,
so these tests delete it to force the second pass.
"""

from __future__ import annotations

import pytest

from smfs_catalog import criteria_migration as _mig
from smfs_catalog import db as _db

from conftest import add_files

DEFAULT = _db.DEFAULT_EXPERIMENTALIST


def _snapshot(db_path: str) -> dict:
    """Every row of every table, except the done-marker's own timestamp."""
    conn = _db.get_connection(db_path)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'")]
        snap = {}
        for t in tables:
            rows = [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")]
            if t == "meta":
                rows = [r for r in rows if r[0] != _mig._MARKER]
            snap[t] = sorted(rows, key=repr)
        return snap
    finally:
        conn.close()


def _alice_catalog(catalog) -> str:
    """Pre-migration catalog: alice owns files, ticked one bound, parked another."""
    add_files(catalog, 2)
    conn = _db.get_connection(catalog)
    with conn:
        conn.execute("UPDATE files SET experimentalist = 'alice'")
    conn.close()
    _db.set_threshold("gating", 1.0, 9.0, "Gating", "alice", catalog)
    _db.set_threshold("parked", 3.0, None, "Parked", "alice", catalog)
    _db.merge_experimentalist_profile(
        "alice", {"criteria_use:gating": 1.0, "other_param": 2.0}, catalog)
    conn = _db.get_connection(catalog)
    with conn:
        conn.execute("DELETE FROM meta WHERE key LIKE 'criteria_%'")
    conn.close()
    return catalog


@pytest.mark.xfail(strict=True, reason=(
    "NOT IDEMPOTENT: a second pass finds the ticks already stripped, computes an "
    "empty legacy gate for alice, and clears her gating bound (1.0, 9.0) to NULL, "
    "overwriting the parked-bounds archive.  D8's retry-next-launch is unsafe "
    "until reconcile can tell an owner was already reconciled."))
def test_second_reconcile_of_an_already_reconciled_owner_changes_nothing(catalog):
    db_path = _alice_catalog(catalog)
    _mig.reconcile(db_path)
    first = _snapshot(db_path)

    conn = _db.get_connection(db_path)       # force the second launch to re-run
    with conn:
        conn.execute("DELETE FROM meta WHERE key = ?", (_mig._MARKER,))
    conn.close()
    _mig.reconcile(db_path)

    assert _snapshot(db_path) == first
