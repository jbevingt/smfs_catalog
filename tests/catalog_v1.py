# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
A catalog written in the version-1 layout, built here and not by the code
under test, so an upgrade test keeps its meaning after db.py changes.

`build_v1(path)` creates the empty tables and stamps `user_version = 1`.
tests/test_invols_catalog.py checks that the layout it makes is the one
recorded for version 1 in tests/test_catalog_version.py (LAYOUT[1]).
"""
import sqlite3

V1_TABLES = (
    """CREATE TABLE files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        filename TEXT NOT NULL,
        size_bytes INTEGER,
        modified_at TEXT,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        spring_constant_pn_nm REAL,
        velocity_nm_s REAL,
        force_dist_nm REAL,
        trigger_point_nn REAL,
        xpos_um REAL,
        ypos_um REAL,
        sample_rate_hz REAL,
        force_filter_bw_hz REAL,
        inv_ols_nm_v REAL,
        microscope_model TEXT,
        measured_date TEXT,
        measured_at TEXT,
        dwell_setting INTEGER,
        indent_mode INTEGER,
        curve_type TEXT,
        unusable_reason TEXT,
        unusable_detail TEXT,
        content_sha256 TEXT,
        parse_ok INTEGER NOT NULL DEFAULT 0,
        parse_error TEXT,
        analyte TEXT,
        technique TEXT,
        solvent TEXT,
        substrate TEXT,
        sample_prep TEXT,
        afm_unit TEXT,
        cantilever TEXT,
        experimentalist TEXT,
        notes TEXT,
        n_traces INTEGER,
        quality TEXT,
        analysis_at TEXT,
        event TEXT,
        primary_segment_idx INTEGER,
        secondary_segment_idx INTEGER,
        segment_override_params_json TEXT,
        tags TEXT DEFAULT '{}')""",
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)",
    """CREATE TABLE thresholds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experimentalist TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        lower_bound REAL,
        upper_bound REAL,
        label TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (experimentalist, analysis_type))""",
    """CREATE TABLE file_metadata (
        file_id INTEGER NOT NULL REFERENCES files(id),
        key TEXT NOT NULL,
        value_real REAL,
        value_text TEXT,
        PRIMARY KEY (file_id, key))""",
    """CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value_real REAL NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE app_settings (
        key TEXT PRIMARY KEY,
        value_text TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE experimentalist_profiles (
        experimentalist TEXT PRIMARY KEY,
        params_json TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE analysis_queue (
        file_id INTEGER PRIMARY KEY REFERENCES files(id),
        enqueued_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending')""",
    """CREATE TABLE analysis_results (
        file_id INTEGER NOT NULL REFERENCES files(id),
        analysis_type TEXT NOT NULL,
        value REAL,
        params_json TEXT NOT NULL DEFAULT '{}',
        code_version TEXT,
        computed_at TEXT NOT NULL,
        PRIMARY KEY (file_id, analysis_type))""",
    """CREATE TABLE event_histograms (
        file_id INTEGER NOT NULL REFERENCES files(id),
        histogram BLOB NOT NULL,
        x_bins INTEGER NOT NULL,
        f_bins INTEGER NOT NULL,
        params_json TEXT NOT NULL DEFAULT '{}',
        computed_at TEXT NOT NULL,
        PRIMARY KEY (file_id))""",
    """CREATE TABLE deflection_histograms (
        file_id INTEGER NOT NULL REFERENCES files(id),
        counts BLOB NOT NULL,
        n_bins INTEGER NOT NULL,
        n_below INTEGER NOT NULL,
        n_above INTEGER NOT NULL,
        params_json TEXT NOT NULL DEFAULT '{}',
        computed_at TEXT NOT NULL,
        PRIMARY KEY (file_id))""",
    """CREATE TABLE event_map (
        file_id INTEGER NOT NULL REFERENCES files(id),
        params_json TEXT NOT NULL DEFAULT '{}',
        code_version TEXT,
        payload_json TEXT NOT NULL,
        computed_at TEXT NOT NULL,
        PRIMARY KEY (file_id, params_json))""",
    """CREATE TABLE distribution_fits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        variable TEXT NOT NULL,
        units TEXT NOT NULL DEFAULT '',
        n_values INTEGER NOT NULL,
        n_peaks INTEGER NOT NULL,
        model_label TEXT NOT NULL,
        params_json TEXT NOT NULL,
        gof_json TEXT NOT NULL,
        fit_config_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        experimentalist TEXT NOT NULL DEFAULT '')""",
    """CREATE TABLE gmm_fits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        x_variable TEXT NOT NULL,
        y_variable TEXT NOT NULL,
        n_values INTEGER NOT NULL,
        k_components INTEGER NOT NULL,
        cov_type TEXT NOT NULL,
        means_json TEXT NOT NULL,
        covs_json TEXT NOT NULL,
        weights_json TEXT NOT NULL,
        gof_json TEXT NOT NULL,
        fit_config_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        experimentalist TEXT NOT NULL DEFAULT '')""",
)


def build_v0(path: str) -> None:
    """Create `path` as an empty catalog from before the version number.

    It has the version-1 tables, less files.substrate and files.sample_prep,
    plus the watched_directories table and files.directory_id (with its index)
    that version 1 drops; `user_version` is 0.
    """
    build_v1(path)
    con = sqlite3.connect(path)
    try:
        with con:
            con.execute("ALTER TABLE files DROP COLUMN substrate")
            con.execute("ALTER TABLE files DROP COLUMN sample_prep")
            con.execute("CREATE TABLE watched_directories "
                        "(id INTEGER PRIMARY KEY, path TEXT)")
            con.execute("ALTER TABLE files ADD COLUMN directory_id INTEGER")
            con.execute("CREATE INDEX idx_files_directory_id "
                        "ON files(directory_id)")
        con.execute("PRAGMA user_version = 0")
    finally:
        con.close()


def build_v1(path: str) -> None:
    """Create `path` as an empty catalog in the version-1 layout."""
    con = sqlite3.connect(path)
    try:
        with con:
            for statement in V1_TABLES:
                con.execute(statement)
        con.execute("PRAGMA user_version = 1")
    finally:
        con.close()
