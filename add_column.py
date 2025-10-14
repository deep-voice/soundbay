#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys

try:
    import pandas as pd
except ImportError:
    print("This script requires pandas. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

def guess_begin_file(txt_path: pathlib.Path) -> str:
    """
    From '7205.230501145122-Raven-....txt' → '7205.230501145122.wav'
    Fallback: if no '-' present, use the stem as-is.
    """
    base = txt_path.name
    m = re.match(r"^([^-\s]+)-", base)
    prefix = m.group(1) if m else txt_path.stem
    return f"{prefix}.wav"

def process_one(
    txt_path: pathlib.Path,
    inplace: bool,
    suffix: str,
    out_dir: pathlib.Path | None,
) -> pathlib.Path:
    begin_file = guess_begin_file(txt_path)

    # Read as strings to preserve format
    df = pd.read_csv(
        txt_path,
        sep="\t",
        dtype=str,
        engine="python",
        encoding="utf-8-sig",
        quoting=3,  # QUOTE_NONE
    )

    # Insert/overwrite "Begin File" as first column
    if "Begin File" not in df.columns:
        df.insert(0, "Begin File", begin_file)
    else:
        df["Begin File"] = begin_file

    # Decide destination
    if inplace:
        out_path = txt_path
    else:
        target_dir = out_dir if out_dir is not None else txt_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        out_name = txt_path.stem + suffix + txt_path.suffix
        out_path = target_dir / out_name

    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    return out_path

def main():
    ap = argparse.ArgumentParser(
        description="Add a 'Begin File' column to Raven-like TSV .txt files."
    )
    ap.add_argument(
        "glob",
        nargs="*",
        default=["*.txt"],
        help="Glob(s) of .txt files to process (default: *.txt)",
    )
    ap.add_argument(
        "-d", "--dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Directory to search (default: current directory)",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the original files instead of creating new ones",
    )
    ap.add_argument(
        "--suffix",
        default="-with-beginfile",
        help="Suffix for output filenames when not using --inplace (default: -with-beginfile)",
    )
    ap.add_argument(
        "-o", "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to write results (created if needed). Ignored with --inplace.",
    )
    args = ap.parse_args()

    if args.inplace and args.out_dir is not None:
        print("Error: --out-dir cannot be used together with --inplace.", file=sys.stderr)
        sys.exit(2)

    # Collect files from provided globs
    folders = [p for p in args.dir.iterdir() if p.is_dir()]
    for folder in folders:
        sub_folder= [p for p in folder.iterdir() if p.is_dir()][0]
        files: list[pathlib.Path] = []
        for g in args.glob:
            files.extend(sorted(sub_folder.glob(g)))

        if not files:
            print("No matching .txt files found.", file=sys.stderr)
            sys.exit(3)

        for p in files:
            if p.is_file() and p.suffix.lower() == ".txt":
                out = process_one(p, inplace=args.inplace, suffix=args.suffix, out_dir=args.out_dir)
                print(f"Wrote: {out}")
            else:
                print(f"Skipped (not a .txt file): {p}", file=sys.stderr)

if __name__ == "__main__":
    main()
