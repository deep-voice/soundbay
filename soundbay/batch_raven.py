#!/usr/bin/env python3
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def inference_csv_to_raven(
    probs_df: pd.DataFrame,
    num_classes: int,
    seq_len: float,
    selected_class: str,
    threshold: float = 0.5,
    class_name: str = "call",
    channel: int = 1,
    max_freq: float = 20_000,
) -> pd.DataFrame:
    """ Converts a pandas DataFrame of inference results into a Raven CSV DataFrame. """
    # figure out which columns hold class probabilities
    prob_cols = probs_df.columns[-int(num_classes):]
    # find segments above threshold
    positive_mask = probs_df[selected_class] > threshold

    # derive begin times
    if "begin_time" in probs_df.columns:
        all_begin = probs_df["begin_time"].values
    else:
        all_begin = np.arange(0, len(probs_df) * seq_len, seq_len)

    begin_times = all_begin[positive_mask]
    end_times = np.round(begin_times + seq_len, 3)

    # cap final end time to file length
    total_duration = round(len(probs_df) * seq_len, 1)
    if len(end_times) and end_times[-1] > total_duration:
        end_times[-1] = total_duration

    # prepare Raven columns
    n = len(begin_times)
    bboxes = {
        "Selection": np.arange(1, n + 1),
        "View": ["Spectrogram 1"] * n,
        "Channel": np.ones(n, dtype=int) * channel,
        "Begin Time (s)": begin_times,
        "End Time (s)": end_times,
        "Low Freq (Hz)": np.zeros(n),
        "High Freq (Hz)": np.ones(n) * max_freq,
        "Annotation": [class_name] * n,
    }

    return pd.DataFrame(bboxes)


def main():
    p = argparse.ArgumentParser(description="Batch-convert inference CSVs to Raven format")
    p.add_argument("--input_dir",  required=True, help="Directory with inference .csv files")
    p.add_argument("--output_dir", required=True, help="Where to save Raven .csv files")
    p.add_argument("--num_classes", type=int, default=2, help="Number of classes in the inference files")
    p.add_argument("--selected_class", default="1", help="Substring to identify the probability column (e.g. '1')")
    p.add_argument("--threshold",     type=float, default=0.5, help="Probability threshold for a positive call")
    p.add_argument("--class_name",    default="call", help="Label to write in the Annotation column")
    p.add_argument("--channel",       type=int,   default=1, help="Channel number for Raven file")
    p.add_argument("--max_freq",      type=float, default=20_000, help="Max frequency for Raven file")
    args = p.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)

        # find the exact probability column name
        prob_col = next(
            (c for c in df.columns if args.selected_class in c), None
        )
        if not prob_col:
            print(f"⚠️  Skipping {csv_path.name}: no column contains '{args.selected_class}'")
            continue

        # get sequence length: from call_length column or default to 0.2s
        seq_len = (
            float(df["call_length"].unique()[0])
            if "call_length" in df.columns
            else 0.2
        )

        raven_df = inference_csv_to_raven(
            probs_df=df,
            num_classes=args.num_classes,
            seq_len=seq_len,
            selected_class=prob_col,
            threshold=args.threshold,
            class_name=args.class_name,
            channel=args.channel,
            max_freq=args.max_freq,
        )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_name  = f"{csv_path.stem}_raven_{timestamp}.csv"
        out_path  = output_dir / out_name

        raven_df.to_csv(out_path, sep="\t", index=False)
        print(f"✔️  {csv_path.name} → {out_name}")

    print("All done.")


if __name__ == "__main__":
    main()
