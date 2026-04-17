"""
Placencia S3 recording coverage and continuity summary.

Goal:
- Given an S3 prefix containing audio files, infer how many days have recordings
  and whether the recordings are continuous, using start time from filenames.

Assumptions (configurable):
- Each recording is 3 hours long, starting at the datetime encoded in the filename.
- Placencia WAV names use a 12-digit block YYMMDDHHMMSS after the device id (e.g. ...201212101523.wav).
- If consecutive start times are ~3 hours apart (within epsilon), treat as continuous.

This script only lists S3 keys (no audio downloads).
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class ParsedKey:
    key: str
    start: dt.datetime  # timezone-aware


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri, allow_fragments=False)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri!r}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(f"Missing bucket in URI: {uri!r}")
    return bucket, prefix


def list_s3_keys(
    *,
    s3_client,
    bucket: str,
    prefix: str,
    suffixes: Sequence[str],
) -> Iterable[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj.get("Key")
            if not key:
                continue
            lower = key.lower()
            if any(lower.endswith(suf) for suf in suffixes):
                yield key


def _basename_stem(key: str) -> str:
    base = os.path.basename(key)
    stem, _ext = os.path.splitext(base)
    return stem


def _try_parse_dt_12digits_yymmddhhmmss(stem: str) -> Optional[dt.datetime]:
    """
    Placencia-style 12 digits after the device id dot: YYMMDDHHMMSS (2-digit year).

    Example stem 67670025.201212101523 -> 2020-12-12 10:15:23
    (not YYYYMMDDHHMM; recordings here are 2020/2021).
    """
    m = re.search(r"(?:^|\.)(?P<dt>\d{12})(?:$|[^\d])", stem)
    if not m:
        return None
    digits = m.group("dt")
    try:
        return dt.datetime.strptime(digits, "%y%m%d%H%M%S")
    except ValueError:
        return None


def _try_parse_dt_15digits_yymmdd_hhmmss_like(stem: str) -> Optional[dt.datetime]:
    """
    Supports common patterns like YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS inside the stem.
    """
    m = re.search(r"(?P<date>\d{8})[_-](?P<time>\d{6})", stem)
    if not m:
        return None
    s = f"{m.group('date')}_{m.group('time')}"
    try:
        return dt.datetime.strptime(s, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _try_parse_dt_14digits_yyyymmddhhmmss(stem: str) -> Optional[dt.datetime]:
    """14 consecutive digits YYYYMMDDHHMMSS anywhere in stem (end-anchored preferred)."""
    m = re.search(r"(?P<dt>\d{14})(?:$|[^\d])", stem)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group("dt"), "%Y%m%d%H%M%S")
    except ValueError:
        return None


# Order matters: first match wins (Placencia 12-digit YY... before delimited / 14-digit fallbacks).
_PARSE_STRATEGIES: List[Tuple[str, Callable[[str], Optional[dt.datetime]]]] = [
    ("yymmddhhmmss_12digit", _try_parse_dt_12digits_yymmddhhmmss),
    ("yyyymmdd_hhmmss_delim", _try_parse_dt_15digits_yymmdd_hhmmss_like),
    ("yyyymmddhhmmss_14digit", _try_parse_dt_14digits_yyyymmddhhmmss),
]


def parse_start_from_key(key: str) -> Tuple[Optional[dt.datetime], Optional[str]]:
    """Return (naive_start, strategy_name) or (None, None)."""
    stem = _basename_stem(key)
    for name, fn in _PARSE_STRATEGIES:
        out = fn(stem)
        if out is not None:
            return out, name
    return None, None


def to_tz_aware(naive: dt.datetime, tz: dt.tzinfo) -> dt.datetime:
    # Treat parsed timestamps as being in the provided timezone.
    return naive.replace(tzinfo=tz)


def iter_daily_overlap_hours(
    start: dt.datetime, end: dt.datetime
) -> Iterable[Tuple[dt.date, float]]:
    """
    Split interval [start, end) into per-calendar-day overlap hours.
    start/end must be timezone-aware in the same tz.
    """
    if end <= start:
        return
    cur = start
    while True:
        day_end = (
            cur.replace(hour=0, minute=0, second=0, microsecond=0)
            + dt.timedelta(days=1)
        )
        seg_end = min(end, day_end)
        hours = (seg_end - cur).total_seconds() / 3600.0
        yield (cur.date(), hours)
        if seg_end >= end:
            break
        cur = seg_end


@dataclass(frozen=True)
class Gap:
    prev_start: dt.datetime
    next_start: dt.datetime
    delta: dt.timedelta


def compute_gaps(
    starts: Sequence[dt.datetime],
    expected_step: dt.timedelta,
    epsilon: dt.timedelta,
) -> list[Gap]:
    gaps: list[Gap] = []
    if len(starts) < 2:
        return gaps
    for a, b in zip(starts, starts[1:]):
        d = b - a
        if abs(d - expected_step) > epsilon:
            gaps.append(Gap(prev_start=a, next_start=b, delta=d))
    return gaps


def longest_continuous_chain(
    starts: Sequence[dt.datetime],
    expected_step: dt.timedelta,
    epsilon: dt.timedelta,
) -> Tuple[int, Optional[dt.datetime], Optional[dt.datetime]]:
    """
    Returns (max_count, chain_start, chain_end).
    Chain end is the last start time included in the chain.
    """
    if not starts:
        return 0, None, None
    if len(starts) == 1:
        return 1, starts[0], starts[0]

    best_n = 1
    best_start = starts[0]
    best_end = starts[0]

    cur_n = 1
    cur_start = starts[0]
    cur_end = starts[0]

    for prev, nxt in zip(starts, starts[1:]):
        d = nxt - prev
        if abs(d - expected_step) <= epsilon:
            cur_n += 1
            cur_end = nxt
        else:
            if cur_n > best_n:
                best_n = cur_n
                best_start = cur_start
                best_end = cur_end
            cur_n = 1
            cur_start = nxt
            cur_end = nxt

    if cur_n > best_n:
        best_n = cur_n
        best_start = cur_start
        best_end = cur_end

    return best_n, best_start, best_end


def _load_tz(tz_name: str) -> dt.tzinfo:
    if tz_name.upper() in ("UTC", "Z"):
        return dt.timezone.utc
    try:
        from zoneinfo import ZoneInfo  # py3.9+
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Timezone support requires Python 3.9+ (zoneinfo). "
            "Use --tz UTC if zoneinfo is unavailable."
        ) from e
    return ZoneInfo(tz_name)


def write_csv(path: str, rows: Sequence[Tuple[dt.date, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "hours"])
        for d, h in rows:
            w.writerow([d.isoformat(), f"{h:.6f}"])


def _month_days_with_recordings(
    hours_by_day: dict[dt.date, float], year: int, month: int
) -> Tuple[int, int, list[dt.date]]:
    """Return (days_in_month, days_with_any_recording, missing_dates_in_month)."""
    last = calendar.monthrange(year, month)[1]
    present = {d for d in hours_by_day if d.year == year and d.month == month}
    missing: list[dt.date] = []
    for day in range(1, last + 1):
        d = dt.date(year, month, day)
        if d not in present:
            missing.append(d)
    return last, len(present), missing


def main(argv: Optional[Sequence[str]] = None) -> int:
    epilog = """
Assumptions (for the biologist summary):
  - Start time is read from each audio filename (see --sample-basenames and strategy counts).
  - Default Placencia pattern: 12 digits YYMMDDHHMMSS after the id (e.g. ...201212101523.wav -> 2020-12-12 10:15:23).
  - Each file contributes a fixed duration (--duration-hours, default 3 h) from that start.
  - Calendar days and overlaps use --tz (default UTC; use America/Belize if timestamps are local).
  - Continuity: consecutive sorted starts are "continuous" if |delta - expected_step| <= epsilon.

Requires: pip install boto3 (or pip install -r requirements.txt) and AWS credentials with
s3:ListBucket on the bucket and list permission on the prefix.
"""
    p = argparse.ArgumentParser(
        description="Compute Placencia recording days, hours/day, and continuity from S3 keys.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--s3-uri",
        required=True,
        help="S3 prefix URI, e.g. s3://bucket/path/to/prefix/",
    )
    p.add_argument(
        "--suffix",
        action="append",
        default=[".wav", ".flac"],
        help="Audio file suffix to include (repeatable). Default: .wav and .flac",
    )
    p.add_argument(
        "--duration-hours",
        type=float,
        default=3.0,
        help="Assumed duration of each recording in hours. Default: 3",
    )
    p.add_argument(
        "--expected-step-hours",
        type=float,
        default=3.0,
        help="Expected time between consecutive starts for continuity. Default: 3",
    )
    p.add_argument(
        "--gap-epsilon-minutes",
        type=float,
        default=15.0,
        help="Tolerance for continuity, in minutes. Default: 15",
    )
    p.add_argument(
        "--tz",
        default="UTC",
        help="Timezone used to interpret filename timestamps and compute calendar days. Default: UTC",
    )
    p.add_argument(
        "--out-csv",
        default="placencia_coverage_by_date.csv",
        help="Output CSV path for hours per day. Default: placencia_coverage_by_date.csv",
    )
    p.add_argument(
        "--max-unmatched-examples",
        type=int,
        default=25,
        help="How many unmatched key examples to print. Default: 25",
    )
    p.add_argument(
        "--sample-basenames",
        type=int,
        default=20,
        help="Print this many sample audio basenames after S3 list (0 to disable). Default: 20",
    )
    p.add_argument(
        "--max-gap-examples",
        type=int,
        default=10,
        help="How many gap examples to print after the biggest gap. Default: 10",
    )

    args = p.parse_args(argv)

    bucket, prefix = parse_s3_uri(args.s3_uri)
    tz = _load_tz(args.tz)

    try:
        import boto3  # type: ignore
    except ModuleNotFoundError:
        print(
            "ERROR: boto3 is not installed in this Python environment.\n"
            "Install it with one of:\n"
            "  - pip install boto3\n"
            "  - pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return 3

    s3 = boto3.client("s3")
    keys = list(
        list_s3_keys(s3_client=s3, bucket=bucket, prefix=prefix, suffixes=args.suffix)
    )

    parsed: list[ParsedKey] = []
    unmatched: list[str] = []
    strategy_counts: Counter[str] = Counter()
    for key in keys:
        naive, strat = parse_start_from_key(key)
        if naive is None:
            unmatched.append(key)
            continue
        strategy_counts[strat or "unknown"] += 1
        parsed.append(ParsedKey(key=key, start=to_tz_aware(naive, tz)))

    parsed.sort(key=lambda x: x.start)
    starts = [x.start for x in parsed]
    start_counts = Counter(starts)
    duplicate_start_timestamps = sum(1 for c in start_counts.values() if c > 1)
    duplicate_extra_files = sum(c - 1 for c in start_counts.values() if c > 1)

    duration = dt.timedelta(hours=float(args.duration_hours))
    expected_step = dt.timedelta(hours=float(args.expected_step_hours))
    epsilon = dt.timedelta(minutes=float(args.gap_epsilon_minutes))

    # Hours per day (overlap split; robust across midnight)
    hours_by_day: dict[dt.date, float] = defaultdict(float)
    for pk in parsed:
        end = pk.start + duration
        for day, hours in iter_daily_overlap_hours(pk.start, end):
            hours_by_day[day] += hours

    days_sorted = sorted(hours_by_day.items(), key=lambda x: x[0])
    write_csv(args.out_csv, days_sorted)

    gaps = compute_gaps(starts, expected_step=expected_step, epsilon=epsilon)
    chain_n, chain_start, chain_end = longest_continuous_chain(
        starts, expected_step=expected_step, epsilon=epsilon
    )

    # Determine coverage date range and missing days within that span.
    missing_days: list[dt.date] = []
    if days_sorted:
        d0 = days_sorted[0][0]
        d1 = days_sorted[-1][0]
        present = set(hours_by_day.keys())
        cur = d0
        while cur <= d1:
            if cur not in present:
                missing_days.append(cur)
            cur += dt.timedelta(days=1)
    else:
        d0 = d1 = None

    print("=== Placencia recording coverage (from S3 keys) ===")
    print(f"S3 URI: {args.s3_uri}")
    print(f"Bucket: {bucket}")
    print(f"Prefix: {prefix}")
    print(f"Timezone (--tz): {args.tz}  (calendar days use this zone)")
    print(f"Assumed duration per file: {args.duration_hours} h")
    print(
        f"Continuity: expected step {args.expected_step_hours} h, "
        f"epsilon ±{args.gap_epsilon_minutes} min"
    )
    print(f"Included suffixes: {args.suffix}")
    print(f"Total matched audio keys: {len(keys)}")
    print(f"Parsed start times: {len(parsed)}")
    if duplicate_start_timestamps:
        print(
            f"Note: {duplicate_start_timestamps} start timestamp(s) appear more than once "
            f"({duplicate_extra_files} extra file(s)); continuity uses sorted starts as-is."
        )
    if strategy_counts:
        print("Filename parse strategy counts (dominant = most common):")
        for name, cnt in strategy_counts.most_common():
            print(f"  {name}: {cnt}")
    if int(args.sample_basenames) > 0 and keys:
        n = min(int(args.sample_basenames), len(keys))
        print("\n--- Sample audio basenames (first keys, listing order) ---")
        for key in keys[:n]:
            print(f"  {os.path.basename(key)}")
        print("(Use strategy counts above to confirm the datetime pattern.)\n")
    print(f"Unmatched keys: {len(unmatched)}")
    if unmatched:
        print("Unmatched examples:")
        for ex in unmatched[: int(args.max_unmatched_examples)]:
            print(f"  - s3://{bucket}/{ex}")

    if d0 is None:
        print("\nNo parsable recording timestamps found. Exiting.")
        return 2

    total_hours = sum(hours_by_day.values())
    print("\n--- Coverage summary ---")
    print(f"Date range (by day): {d0.isoformat()} .. {d1.isoformat()}")
    print(f"Days with recordings: {len(hours_by_day)}")
    print(f"Total assumed recorded hours: {total_hours:.2f}")
    print(f"Wrote hours/day CSV: {args.out_csv}")
    if missing_days:
        print(f"Missing days within range: {len(missing_days)}")
        print("Missing day examples:")
        for md in missing_days[:25]:
            print(f"  - {md.isoformat()}")
    else:
        print("Missing days within range: 0 (every day in span has at least one file)")

    # Per-calendar-month coverage (any month that has at least one recording day)
    months_seen = sorted({(d.year, d.month) for d in hours_by_day})
    if months_seen:
        print("\n--- Per-calendar-month coverage ---")
        print(
            "(Days with recordings vs days in month; missing = no overlap with any "
            f"{args.duration_hours} h window starting at a parsed filename time.)"
        )
        for y, m in months_seen:
            dim, with_rec, miss = _month_days_with_recordings(hours_by_day, y, m)
            print(
                f"  {y:04d}-{m:02d}: {with_rec}/{dim} days with recordings; "
                f"missing {len(miss)} day(s) in month"
            )
            if miss and len(miss) <= 10:
                print(f"    missing: {', '.join(d.isoformat() for d in miss)}")
            elif miss:
                print(
                    f"    missing (first 10): {', '.join(x.isoformat() for x in miss[:10])}..."
                )

    print("\n--- Continuity summary (start-time spacing) ---")
    print(
        "Continuity rule: consecutive starts are continuous if "
        f"|delta - {expected_step}| <= {epsilon}"
    )
    if starts:
        print(f"Earliest start: {starts[0].isoformat()}")
        print(f"Latest start: {starts[-1].isoformat()}")
    print(f"Gap count (outside tolerance): {len(gaps)}")
    if gaps:
        biggest = max(gaps, key=lambda g: g.delta)
        print(
            "Biggest gap: "
            f"{biggest.prev_start.isoformat()} -> {biggest.next_start.isoformat()} "
            f"(delta={biggest.delta})"
        )
        k = int(args.max_gap_examples)
        if k > 0:
            print(f"First {min(k, len(gaps))} gap(s) (outside tolerance):")
            for g in gaps[:k]:
                print(
                    f"  {g.prev_start.isoformat()} -> {g.next_start.isoformat()} "
                    f"delta={g.delta}"
                )
    if chain_start and chain_end:
        chain_span = (chain_end - chain_start) + expected_step
        print(
            f"Longest continuous chain: {chain_n} files "
            f"from {chain_start.isoformat()} to {chain_end.isoformat()} "
            f"(~{chain_span.total_seconds()/3600.0:.1f} hours assuming step)"
        )
    else:
        print("Longest continuous chain: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

