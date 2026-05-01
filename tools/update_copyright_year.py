#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-commit hook: ensures SPDX copyright year covers the current year.

Auto-updates copyright year to include the current year, e.g.:
  before: FileCopyrightText: Copyright (c) <START> NVIDIA ...
  after:  FileCopyrightText: Copyright (c) <START>-<NOW> NVIDIA ...
Modified files are rewritten in place;
the hook exits non-zero so pre-commit asks the user to re-stage them.
"""

import re
import sys
from datetime import date
from pathlib import Path

CURRENT_YEAR = str(date.today().year)

# Matches the year field inside any SPDX-FileCopyrightText line.
_SPDX_YEAR_RE = re.compile(
    r"(SPDX-FileCopyrightText:.*?Copyright\s+\(c\)\s+)" r"(\d{4})" r"(-\d{4})?"
)


def _replace(m: re.Match) -> str:
    prefix, start, end = m.group(1), m.group(2), m.group(3)
    last = end[1:] if end else start
    if last == CURRENT_YEAR:
        return m.group(0)
    if start == CURRENT_YEAR:
        return m.group(0)
    return f"{prefix}{start}-{CURRENT_YEAR}"


def update_file(filepath: str) -> bool:
    path = Path(filepath)
    try:
        original = path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"{filepath}: cannot read: {exc}", file=sys.stderr)
        return False

    updated = _SPDX_YEAR_RE.sub(_replace, original)
    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")
    print(f"{filepath}: updated copyright year to {CURRENT_YEAR}")
    return True


def main() -> int:
    # Evaluate all files — do NOT use any() which short-circuits and skips
    # remaining files once the first modification is found.
    results = [update_file(f) for f in sys.argv[1:]]
    return 1 if any(results) else 0


if __name__ == "__main__":
    sys.exit(main())
