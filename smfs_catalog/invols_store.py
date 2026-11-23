"""Everything the invOLS window and dashboard need from the catalog: the trace
table, applying an ensemble value, reverting to the own fit, source labels."""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from . import db, quantities, variables
from .curve_analysis import pipeline_params_from
from .invols_stats import (
    AVERAGES, Summary, applied_value, own_transform_nm_v, summarise,
    usable_instrument,
)
from .provenance import cache_version

TRANSFORM_KEY = "transform_invols_nm_v"
_AVERAGE_SETTING = "invols_average"
_METHODS = ("all", "box")
_FIT_SETTING_KEYS = ("invols_offset_pts", "invols_window_pts", "spectral_cutoff_hz")


@dataclass(frozen=True)
class TraceRow:
    file_id: int
    path: str
    time_s: float | None
    instrument_nm_v: float | None
    own_nm_v: float | None
    fit_current: bool
    rms_nm: float | None
    source: str
    group_id: int | None
    in_use_nm_v: float | None


@dataclass(frozen=True)
class EnsembleRequest:
    member_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    method: str
    box: tuple[float, float, float, float] | None
    fit_settings: Mapping[str, float]
    average: str = "arithmetic"


@dataclass(frozen=True)
class ApplyResult:
    group_id: int | None
    summary: Summary
    n_written: int
    skipped_no_instrument: tuple[int, ...]
    left_out_old_settings: tuple[int, ...]


class InvolsApplyError(ValueError):
    pass


@dataclass(frozen=True)
class _Group:
    group_id: int
    value: float
    n: int
    created_at: str
    fit_settings_json: str


# ── Reading ───────────────────────────────────────────────────────────────────

def _fetch_in(conn: sqlite3.Connection, sql: str, ids: Sequence[int]) -> list:
    """Rows of `sql` (one `{ph}` placeholder group) for `ids`, chunked."""
    rows: list = []
    for chunk in db._sql_chunks(list(ids)):
        ph = ",".join("?" * len(chunk))
        rows += conn.execute(sql.format(ph=ph), chunk).fetchall()
    return rows


def _results(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, dict[str, tuple]]:
    """{file_id: {'invols_slope'|'invols_rms': (value, params_json)}}."""
    rows = _fetch_in(conn, """
        SELECT file_id, analysis_type, value, params_json FROM analysis_results
        WHERE analysis_type IN ('invols_slope', 'invols_rms')
          AND file_id IN ({ph})""", ids)
    out: dict[int, dict[str, tuple]] = {}
    for r in rows:
        out.setdefault(r["file_id"], {})[r["analysis_type"]] = (r["value"], r["params_json"])
    return out


def _ensemble_groups(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, _Group]:
    """The group in force for each file that has one.  A file whose instrument
    value is unusable has no ensemble in force, whatever its assignment row says."""
    in_force = db.ensemble_divisors(list(ids), conn=conn)
    rows = _fetch_in(conn, """
        SELECT a.file_id, g.id, g.transform_nm_v, g.n, g.created_at, g.fit_settings_json
        FROM invols_assignments a JOIN invols_groups g ON g.id = a.group_id
        WHERE a.file_id IN ({ph})""", ids)
    return {r["file_id"]: _Group(r["id"], r["transform_nm_v"], r["n"],
                                 r["created_at"], r["fit_settings_json"])
            for r in rows if r["file_id"] in in_force}


def _own_fit(instrument: float | None, found: dict[str, tuple]) -> tuple:
    """(own nm/V or None, slope row's params_json or None) for one file."""
    slope_row = found.get("invols_slope")
    if slope_row is None:
        return None, None
    return own_transform_nm_v(instrument, slope_row[0]), slope_row[1]


def _fit_key(db_path: str) -> str:
    return pipeline_params_from(db.load_analysis_params(db_path)).params_invols


def _trace_row(f: sqlite3.Row, found: dict[str, tuple], group: _Group | None,
               time_s: float | None, key: str) -> TraceRow:
    own, stored_key = _own_fit(f["inv_ols_nm_v"], found)
    rms = found.get("invols_rms", (None,))[0]
    return TraceRow(
        f["id"], f["path"], time_s, f["inv_ols_nm_v"], own,
        own is not None and stored_key == key, rms,
        "own" if group is None else "ensemble",
        None if group is None else group.group_id,
        own if group is None else group.value)


def trace_rows(paths: Sequence[str], db_path: str) -> list[TraceRow]:
    key = _fit_key(db_path)
    times = variables.values(list(paths), [variables.TIME_KEY], db_path)
    conn = db.get_connection(db_path)
    try:
        files = _fetch_in(conn, "SELECT id, path, inv_ols_nm_v FROM files "
                          "WHERE path IN ({ph})", [db.normalize_path(p) for p in paths])
        ids = [f["id"] for f in files]
        found, groups = _results(conn, ids), _ensemble_groups(conn, ids)
    finally:
        conn.close()
    rows = []
    for f in files:
        t = times.get(f["path"], {}).get(variables.TIME_KEY)
        t = None if t is None or not math.isfinite(t) else t
        rows.append(_trace_row(f, found.get(f["id"], {}), groups.get(f["id"]), t, key))
    rows.sort(key=lambda r: (math.inf if r.time_s is None else r.time_s, r.path))
    return rows


def fit_settings_now(db_path: str) -> dict[str, float]:
    params = db.load_analysis_params(db_path)
    return {k: float(getattr(params, k)) for k in _FIT_SETTING_KEYS}


def stale_setting_groups(paths: Sequence[str], db_path: str) -> list[int]:
    now = fit_settings_now(db_path)
    conn = db.get_connection(db_path)
    try:
        ids = [r["id"] for r in _fetch_in(
            conn, "SELECT id FROM files WHERE path IN ({ph})",
            [db.normalize_path(p) for p in paths])]
        groups = _ensemble_groups(conn, ids)
    finally:
        conn.close()
    stale = {g.group_id for g in groups.values()
             if json.loads(g.fit_settings_json) != now}
    return sorted(stale)


def get_average(db_path: str) -> str:
    stored = db.get_app_setting(_AVERAGE_SETTING, AVERAGES[0], db_path)
    return stored if stored in AVERAGES else AVERAGES[0]


def set_average(average: str, db_path: str) -> None:
    if average not in AVERAGES:
        raise ValueError(f"unknown average: {average!r}")
    db.set_app_setting(_AVERAGE_SETTING, average, db_path)


def _group_label(group: _Group | None) -> str:
    if group is None:
        return "own fit"
    return f"ensemble of {group.n} traces, {group.created_at[:10]}"


def source_labels(file_ids: Sequence[int], db_path: str) -> dict[int, str]:
    conn = db.get_connection(db_path)
    try:
        groups = _ensemble_groups(conn, file_ids)
    finally:
        conn.close()
    return {fid: _group_label(groups.get(fid)) for fid in file_ids}


def _nm_v(value: float | None) -> str:
    if value is None:
        return "—"
    return quantities.format_value(TRANSFORM_KEY, value, with_unit=True)


def curve_label(path: str, own_slope: float | None,
                instrument_nm_v: float | None, db_path: str) -> str:
    conn = db.get_connection(db_path)
    try:
        fid = db.get_file_id(path, conn=conn)
        ids = [] if fid is None else [fid]
        group = _ensemble_groups(conn, ids).get(fid)
        rms = _results(conn, ids).get(fid, {}).get("invols_rms", (None,))[0]
    finally:
        conn.close()
    own = own_transform_nm_v(instrument_nm_v, own_slope)
    if group is None:
        parts = [f"InvOLS in use {_nm_v(own)} (own fit)"]
    else:
        parts = [f"InvOLS in use {_nm_v(group.value)} (ensemble of {group.n} traces)",
                 f"own fit {_nm_v(own)}"]
    if own_slope is not None and math.isfinite(own_slope):
        parts.append("slope=" + quantities.format_value("invols_slope", own_slope))
    if rms is not None and math.isfinite(rms):
        parts.append("RMS=" + quantities.format_value("invols_rms", rms, with_unit=True))
    return " · ".join(parts)


# ── Writing ───────────────────────────────────────────────────────────────────

@contextmanager
def _write_transaction(db_path: str):
    """One BEGIN IMMEDIATE ... COMMIT; any failure rolls everything back.  The
    connection is taken out of the driver's implicit transaction handling so
    nothing commits part-way."""
    conn = db.get_connection(db_path)
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _validate(request: EnsembleRequest) -> None:
    if request.method not in _METHODS:
        raise ValueError(f"unknown method: {request.method!r}")
    if request.method == "box" and request.box is None:
        raise ValueError("method 'box' needs a box")
    if request.average not in AVERAGES:
        raise ValueError(f"unknown average: {request.average!r}")


def _instruments(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, float | None]:
    rows = _fetch_in(conn, "SELECT id, inv_ols_nm_v FROM files WHERE id IN ({ph})", ids)
    return {r["id"]: r["inv_ols_nm_v"] for r in rows}


def _eligible_members(conn: sqlite3.Connection, member_ids: Sequence[int],
                      key: str) -> tuple[list[tuple[int, float]], tuple[int, ...]]:
    """(members with a current own value, members left out for an older fit)."""
    instruments = _instruments(conn, member_ids)
    found = _results(conn, member_ids)
    used: list[tuple[int, float]] = []
    left_out: list[int] = []
    for fid in member_ids:
        own, stored_key = _own_fit(instruments.get(fid), found.get(fid, {}))
        if own is None:
            continue
        if stored_key == key:
            used.append((fid, own))
        else:
            left_out.append(fid)
    return used, tuple(left_out)


def _insert_group(conn: sqlite3.Connection, request: EnsembleRequest,
                  summary: Summary, value: float, settings: Mapping[str, float]) -> int:
    box = None if request.method != "box" else json.dumps([float(x) for x in request.box])
    cur = conn.execute("""
        INSERT INTO invols_groups (created_at, method, n, average, transform_nm_v,
                                   sd_nm_v, se_nm_v, box_json, fit_settings_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(timespec="seconds"), request.method, summary.n,
         request.average, value, summary.sd, summary.se, box, json.dumps(dict(settings))))
    return cur.lastrowid


def _assign_targets(conn: sqlite3.Connection, group_id: int,
                    targets: Sequence[int], value: float) -> tuple[int, tuple[int, ...]]:
    version = cache_version()
    if version is None:
        raise InvolsApplyError("no code identity: results cannot be stored")
    instruments = _instruments(conn, targets)
    skipped = tuple(t for t in targets if not usable_instrument(instruments.get(t)))
    written = [t for t in targets if t in instruments and t not in skipped]
    for fid in written:
        conn.execute("INSERT OR REPLACE INTO invols_assignments (file_id, group_id) "
                     "VALUES (?, ?)", (fid, group_id))
        db.write_analysis_results_multi(
            fid, {TRANSFORM_KEY: value}, json.dumps({"invols_group": group_id}),
            version, conn=conn, commit=False)
    return len(written), skipped


def apply_ensemble(request: EnsembleRequest, db_path: str) -> ApplyResult:
    _validate(request)
    settings = fit_settings_now(db_path)
    key = _fit_key(db_path)
    members = tuple(dict.fromkeys(request.member_ids))
    targets = tuple(dict.fromkeys(request.target_ids))
    with _write_transaction(db_path) as conn:
        used, left_out = _eligible_members(conn, members, key)
        summary = summarise(v for _, v in used)
        value = applied_value(summary, request.average) if used else None
        if value is None:
            raise InvolsApplyError("no trace with a current own fit can give a value")
        group_id = _insert_group(conn, request, summary, value, settings)
        conn.executemany("INSERT INTO invols_group_members (group_id, file_id) "
                         "VALUES (?, ?)", [(group_id, fid) for fid, _ in used])
        n_written, skipped = _assign_targets(conn, group_id, targets, value)
    return ApplyResult(group_id, summary, n_written, skipped, left_out)


def apply_own(target_ids: Sequence[int], db_path: str) -> int:
    key = _fit_key(db_path)
    version = cache_version()
    if version is None:
        raise InvolsApplyError("no code identity: results cannot be stored")
    targets = list(dict.fromkeys(target_ids))
    with _write_transaction(db_path) as conn:
        instruments = _instruments(conn, targets)
        found = _results(conn, targets)
        present = [t for t in targets if t in instruments]
        for fid in present:
            own, stored_key = _own_fit(instruments[fid], found.get(fid, {}))
            conn.execute("DELETE FROM invols_assignments WHERE file_id = ?", (fid,))
            db.write_analysis_results_multi(
                fid, {TRANSFORM_KEY: own}, stored_key or key, version,
                conn=conn, commit=False)
    return len(present)
