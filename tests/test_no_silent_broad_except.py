# Copyright (C) 2026 Joseph Hamill
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See the LICENSE file in the
# repository root, or <https://www.gnu.org/licenses/>.

"""
Guard test: no broad `except` swallows an error silently.

WHY THIS EXISTS.  Around twenty handlers of the form

    except Exception:
        return {}

were scattered through the windows.  Each turned a real failure into something
that looked like a normal empty state:

  * a failed freshness check showed "nothing stale";
  * a failed curve load showed "No curve";
  * a failed multi-event computation said "failed" with no error text;
  * a corrupt profile looked like "no experimentalist".

None of it reached the user or crash.log, so the first symptom was a wrong
screen and the cause was unrecoverable.  Unit tests cannot reach this in a GUI
app (see test_names_are_imported.py for why), so this guard is source-level.

THE RULE.  A handler is flagged when it catches Exception or BaseException
and does none of:

  1. re-raise: contains a `raise` anywhere in its body (bare `raise`, or
     raising a different error, which is still visible);
  2. use the caught exception: it binds `as exc` AND `exc` is read in the
     body (passed to report_handled / log_handled, formatted, returned...).
     `except Exception as exc: pass` binds but never uses it, so it is flagged;
  3. carry a reason on its `except` line:
         except Exception:  # noqa: BLE001  shutdown, nothing to report to
     The code BLE001 must be followed by a non-empty reason.  A bare
     `# noqa: BLE001` is flagged: the reason is the point.

WHAT COUNTS AS "BROAD".  `except Exception`, `except BaseException`, a bare
`except:`, and tuples containing either (`except (OSError, Exception)` is as
broad as `except Exception`).  Narrow handlers (`except ValueError`) are never
flagged: they are the fix, not the problem.  Only plain names are recognised
(`Exception`, not `builtins.Exception`); nobody writes the latter here.

Scope: every module in smfs_catalog/ and scripts/.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BROAD = {"Exception", "BaseException"}

# `# noqa: BLE001` then whitespace then at least one non-space reason char.
NOQA_WITH_REASON = re.compile(r"#\s*noqa:\s*BLE001\s+\S")


def _modules():
    return sorted(list((ROOT / "smfs_catalog").glob("*.py"))
                  + list((ROOT / "scripts").glob("*.py")))


def _is_broad(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:  # bare `except:`
        return True
    names = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(n, ast.Name) and n.id in BROAD for n in names)


def _reraises(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(n, ast.Raise)
               for stmt in handler.body for n in ast.walk(stmt))


def _uses_exception(handler: ast.ExceptHandler) -> bool:
    if not handler.name:
        return False
    return any(isinstance(n, ast.Name) and n.id == handler.name
               and isinstance(n.ctx, ast.Load)
               for stmt in handler.body for n in ast.walk(stmt))


def _is_visible(handler: ast.ExceptHandler, lines: list[str]) -> bool:
    return (_reraises(handler) or _uses_exception(handler)
            or bool(NOQA_WITH_REASON.search(lines[handler.lineno - 1])))


class _Finder(ast.NodeVisitor):
    def __init__(self, lines: list[str]):
        self.lines = lines
        self.scope: list[str] = []
        self.flagged: list[tuple[int, str]] = []

    def _enter(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _enter

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if _is_broad(node) and not _is_visible(node, self.lines):
            self.flagged.append((node.lineno, ".".join(self.scope) or "<module>"))
        self.generic_visit(node)


def silent_broad_handlers(source: str) -> list[tuple[int, str]]:
    """(line, enclosing function) of every silent broad handler in `source`."""
    finder = _Finder(source.splitlines())
    finder.visit(ast.parse(source))
    return finder.flagged


@pytest.mark.parametrize("path", _modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_silent_broad_except_in_module(path: Path):
    flagged = silent_broad_handlers(path.read_text(encoding="utf-8"))
    listing = "\n".join(f"  {path.name}:{ln} in {fn}" for ln, fn in flagged)
    assert not flagged, (
        f"silent broad except handler(s):\n{listing}\n"
        "Surface it (qt_utils.report_handled), log it (crashlog.log_handled), "
        "narrow it to the errors you expect, or mark the except line "
        "`# noqa: BLE001  <reason>`."
    )


# --------------------------------------------------------------------------
# Self-tests: the checker itself, on small source strings.
# --------------------------------------------------------------------------

def _wrap(handler_src: str) -> str:
    body = "\n".join("    " + ln for ln in handler_src.splitlines())
    return f"def f():\n    try:\n        g()\n{body}\n"


def test_the_guard_would_have_caught_the_bug_it_was_written_for():
    """The original defect: swallow everything, return an empty value."""
    src = "class W:\n    def freshness(self):\n        try:\n            g()\n" \
          "        except Exception:\n            return {}\n"
    assert silent_broad_handlers(src) == [(5, "W.freshness")]


@pytest.mark.parametrize("handler", [
    "except Exception:\n    return {}",
    "except BaseException:\n    pass",
    "except:\n    pass",
    "except (OSError, Exception):\n    pass",
    "except Exception as exc:\n    pass",                 # binds, never uses
    "except Exception:  # noqa: BLE001\n    pass",        # no reason
    "except Exception:  # noqa: BLE001   \n    pass",     # blank reason
    "except Exception:  # noqa: E722  shutdown\n    pass",  # wrong code
    "except Exception:\n    # noqa: BLE001  reason on the wrong line\n    pass",
])
def test_flags_silent_handlers(handler):
    assert len(silent_broad_handlers(_wrap(handler))) == 1


@pytest.mark.parametrize("handler", [
    "except Exception as exc:\n    report_handled(self._lbl.setText, 'Freshness', exc)\n    return {}",
    "except Exception as exc:\n    log_handled('x', exc)",
    "except Exception:\n    raise",
    "except Exception as exc:\n    raise RuntimeError('x') from exc",
    "except BaseException:  # noqa: BLE001  shutdown\n    pass",
    "except Exception:  # noqa: BLE001  parked: D8 awaiting decision\n    pass",
    "except:  # noqa: BLE001  last resort\n    pass",
    "except ValueError:\n    pass",                        # narrow, not broad
    "except (OSError, ValueError):\n    return None",      # narrow tuple
])
def test_passes_visible_or_justified_handlers(handler):
    assert silent_broad_handlers(_wrap(handler)) == []


def test_reports_each_handler_with_its_line_and_function():
    src = (
        "def a():\n"
        "    try:\n        g()\n"
        "    except Exception:\n        pass\n"
        "def b():\n"
        "    try:\n        g()\n"
        "    except Exception as e:\n        print(e)\n"
        "    try:\n        g()\n"
        "    except BaseException:\n        pass\n"
    )
    assert silent_broad_handlers(src) == [(4, "a"), (13, "b")]
