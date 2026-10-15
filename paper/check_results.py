"""Check that make_data.py emits exactly the numbers in the report.

    python paper/make_data.py              # regenerate the report's tables and results
    python paper/check_results.py          # compare them with paper/results.sha256
    python paper/check_results.py --write  # record new digests after an intended change

Each LaTeX table is hashed byte for byte. Each JSON file is hashed after rounding every float to
ten significant digits: finer than any number the report prints, and coarser than the last-bit
differences between BLAS builds (a least-squares fit can differ in its sixteenth digit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "results.sha256"
OUTPUTS = ("results.json", "numbers.json", "site.json", "tables/*.tex")  # what make_data.py writes, relative to paper/
DIGITS = 10


def rounded(x):
    if isinstance(x, float):
        return float(f"{x:.{DIGITS}g}")
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    return x


def digest(path: Path) -> str:
    if path.suffix == ".json":
        blob = json.dumps(rounded(json.loads(path.read_text())), separators=(",", ":")).encode()
    else:
        blob = path.read_bytes()
    return hashlib.sha256(blob).hexdigest()


def current() -> dict[str, str]:
    files = sorted({p for pattern in OUTPUTS for p in HERE.glob(pattern)})
    return {p.relative_to(HERE).as_posix(): digest(p) for p in files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="record the current digests in results.sha256")
    args = ap.parse_args(argv)
    now = current()
    if not now:
        print("no outputs found: run python paper/make_data.py first", file=sys.stderr)
        return 1
    if args.write:
        RECORD.write_text("".join(f"{h}  {name}\n" for name, h in now.items()))
        print(f"recorded {len(now)} digests in {RECORD.name}")
        return 0
    expected = {name: h for h, name in (line.split(maxsplit=1) for line in RECORD.read_text().splitlines() if line.strip())}
    bad = [f"missing  {name}" for name in expected if name not in now]
    bad += [f"changed  {name}" for name in expected if name in now and now[name] != expected[name]]
    bad += [f"new      {name}" for name in now if name not in expected]
    for line in bad:
        print(line)
    if bad:
        print(f"{len(bad)} of {len(expected)} outputs differ from {RECORD.name}", file=sys.stderr)
        return 1
    print(f"all {len(expected)} outputs match {RECORD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
