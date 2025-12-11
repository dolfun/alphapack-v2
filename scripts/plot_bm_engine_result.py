import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
  parser = argparse.ArgumentParser(description="Plot benchmark stats vs batch size (mean ± std).")
  parser.add_argument(
    "csv_path",
    type=str,
    help="Path to benchmark_results.csv",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="benchmark_summary.png",
    help="Output PNG filename (default: benchmark_summary.png)",
  )
  return parser.parse_args()


def main():
  args = parse_args()

  df = pd.read_csv(args.csv_path)

  grouped = df.groupby("batch_size")
  batch_sizes = sorted(grouped.groups.keys())  # type: ignore

  mean_batch_tp = grouped["batch_throughput_batches_per_sec"].mean()
  std_batch_tp = grouped["batch_throughput_batches_per_sec"].std()

  mean_batch_lat = grouped["batch_latency_avg_ms"].mean()
  std_batch_lat = grouped["batch_latency_avg_ms"].std()

  mean_single_tp = grouped["single_throughput_evals_per_sec"].mean()
  std_single_tp = grouped["single_throughput_evals_per_sec"].std()

  mean_single_lat = grouped["single_latency_avg_ms"].mean()
  std_single_lat = grouped["single_latency_avg_ms"].std()

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  axes = axes.flatten()

  def style_axis(ax):
    ax.set_xlabel("Batch Size")
    ax.set_xticks(batch_sizes)
    ax.grid(True, alpha=0.3)

  axes[0].errorbar(batch_sizes, mean_batch_tp, yerr=std_batch_tp, marker="o")
  axes[0].set_title("Batch Throughput vs Batch Size (mean ± std)")
  axes[0].set_ylabel("Batch Throughput (batches/sec)")
  style_axis(axes[0])

  axes[1].errorbar(batch_sizes, mean_batch_lat, yerr=std_batch_lat, marker="o")
  axes[1].set_title("Batch Latency vs Batch Size (mean ± std)")
  axes[1].set_ylabel("Batch Latency (ms)")
  style_axis(axes[1])

  axes[2].errorbar(batch_sizes, mean_single_tp, yerr=std_single_tp, marker="o")
  axes[2].set_title("Single-Eval Throughput vs Batch Size (mean ± std)")
  axes[2].set_ylabel("Single-Eval Throughput (evals/sec)")
  style_axis(axes[2])

  axes[3].errorbar(batch_sizes, mean_single_lat, yerr=std_single_lat, marker="o")
  axes[3].set_title("Single-Eval Latency vs Batch Size (mean ± std)")
  axes[3].set_ylabel("Single-Eval Latency (ms)")
  style_axis(axes[3])

  fig.tight_layout()
  fig.savefig(args.output, dpi=200)
  print(f"Saved plot to {args.output}")


if __name__ == "__main__":
  main()
