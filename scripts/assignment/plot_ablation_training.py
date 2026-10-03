"""Plot Ant training statistics, not OOD evaluation scores.

Accept a raw/summary CSV or native run folders. Raw curves are the default.
--smoothing-window N overlays a trailing rolling mean, explicitly labelled, with
raw curves still visible. Input data/event files are never modified.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_ablation_training import scalar_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", type=Path)
    inputs.add_argument("--run-folders", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be new to avoid overwriting figures.")
    parser.add_argument("--smoothing-window", type=int, default=1)
    args = parser.parse_args()
    if args.smoothing_window < 1:
        parser.error("Smoothing window must be positive.")
    values = {}
    metrics = {"mean_reward": "Mean training episode return", "episode_length": "Mean training episode length (steps)"}
    if args.csv:
        with args.csv.open(newline="") as stream:
            for row in csv.DictReader(stream):
                run = values.setdefault(row["run"], {metric: {} for metric in metrics})
                for metric in metrics:
                    if row.get(metric) not in (None, "", "NaN"):
                        # Summary CSV explicitly records the source iteration of nearest samples.
                        step = int(row.get(metric + "_source_step") or row["iteration"])
                        run[metric][step] = float(row[metric])
    else:
        for directory in args.run_folders:
            _, _, data, _ = scalar_data(directory)
            values[directory.name] = {metric: data[metric] for metric in metrics}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for metric, ylabel in metrics.items():
        figure, axis = plt.subplots(figsize=(10, 5))
        for name, series in values.items():
            steps = sorted(series)
            if not steps:
                continue
            raw = [series[step] for step in steps]
            line, = axis.plot(steps, raw, alpha=0.25 if args.smoothing_window > 1 else 1.0,
                              label=f"{name} (raw)")
            if args.smoothing_window > 1:
                window = args.smoothing_window
                smoothed = [sum(raw[max(0, index-window+1):index+1]) / min(window, index+1)
                            for index in range(len(raw))]
                axis.plot(steps, smoothed, color=line.get_color(), label=f"{name} (rolling mean, N={window})")
        axis.set(xlabel="PPO iteration", ylabel=ylabel, title="Training statistics — not unseen evaluation")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{metric}.png", dpi=150)
        plt.close(figure)


if __name__ == "__main__":
    main()
