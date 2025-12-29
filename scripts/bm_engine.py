import argparse
import csv
import subprocess
import sys
import re
import itertools
from pathlib import Path

BATCH_RESULTS_RE = re.compile(
  r"Results \(Batch Size:\s*(\d+)\):\s*"
  r"\n\s*Throughput:\s*([0-9.]+)\s+batches/sec"
  r"\n\s*Time Taken:\s*([0-9.]+)\s+sec"
  r"\n\s*Batch Latency:\s*"
  r"\n\s*Avg:\s*([0-9.]+)\s+ms"
  r"\n\s*Min:\s*([0-9.]+)\s+ms"
  r"\n\s*Max:\s*([0-9.]+)\s+ms"
  r"\n\s*Std Dev:\s*([0-9.]+)\s+ms"
  r"\n\s*In-Flight Batches:\s*"
  r"\n\s*Avg\(Meas\.\):\s*([0-9.]+)"
  r"\n\s*Avg\(Calc\.\):\s*([0-9.]+)"
  r"\n\s*Max:\s*([0-9.]+)", re.MULTILINE
)

SINGLE_RESULTS_RE = re.compile(
  r"Results \(Single Evaluation\):\s*"
  r"\n\s*Throughput:\s*([0-9.]+)\s+evals/sec"
  r"\n\s*Latency:\s*"
  r"\n\s*Avg:\s*([0-9.]+)\s+ms", re.MULTILINE
)


def parse_output(output: str, expected_batch_size: int):
  batch_match = BATCH_RESULTS_RE.search(output)
  if not batch_match:
    raise RuntimeError("Missing batch-size results section or format mismatch.")

  single_match = SINGLE_RESULTS_RE.search(output)
  if not single_match:
    raise RuntimeError("Missing single-evaluation results section or format mismatch.")

  batch_size = int(batch_match.group(1))
  if batch_size != expected_batch_size:
    raise RuntimeError(f"Expected batch {expected_batch_size}, got {batch_size}")

  return {
    "batch_size": batch_size,
    "batch_throughput_batches_per_sec": float(batch_match.group(2)),
    "time_taken_sec": float(batch_match.group(3)),
    "batch_latency_avg_ms": float(batch_match.group(4)),
    "batch_latency_min_ms": float(batch_match.group(5)),
    "batch_latency_max_ms": float(batch_match.group(6)),
    "batch_latency_std_ms": float(batch_match.group(7)),
    "avg_in_flight_measured": float(batch_match.group(8)),
    "avg_in_flight_calculated": float(batch_match.group(9)),
    "max_in_flight": float(batch_match.group(10)),
    "single_throughput_evals_per_sec": float(single_match.group(1)),
    "single_latency_avg_ms": float(single_match.group(2)),
  }


def run_benchmark(
  exe_path: Path, model_path: Path, batch_size: int, threads: int, streams: int
) -> str:
  cmd = [str(exe_path), str(model_path), str(batch_size), str(threads), str(streams)]
  print(f"Running: {' '.join(cmd)}")

  result = subprocess.run(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
  )

  if result.returncode != 0:
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    raise RuntimeError(f"Benchmark failed with return code {result.returncode}")

  return result.stdout


def main():
  parser = argparse.ArgumentParser(description="Grid search benchmark script.")
  parser.add_argument("--exe", required=True, help="Path to the C++ executable")
  parser.add_argument("--model", required=True, help="Path to the model file")

  parser.add_argument(
    "--batch_sizes", type=int, nargs='+', required=True, help="List of batch sizes (e.g. 16 32 64)"
  )
  parser.add_argument("--threads", type=int, nargs='+', required=True, help="List of thread counts")
  parser.add_argument("--streams", type=int, nargs='+', required=True, help="List of stream counts")

  parser.add_argument(
    "--runs", type=int, required=True, help="Number of times to repeat each combination"
  )
  parser.add_argument("--output", default="benchmark_results.csv", help="Output CSV file path")

  args = parser.parse_args()

  exe_path = Path(args.exe)
  model_path = Path(args.model)
  all_rows = []

  combinations = list(itertools.product(args.batch_sizes, args.threads, args.streams))
  total_tasks = len(combinations) * args.runs

  print(f"Total Unique Combinations: {len(combinations)}")
  print(f"Total Benchmark Runs: {total_tasks}")

  for run in range(1, args.runs + 1):
    print(f"\n--- STARTING RUN {run}/{args.runs} ---")

    for batch, threads, streams in combinations:
      try:
        output = run_benchmark(exe_path, model_path, batch, threads, streams)
        metrics = parse_output(output, batch)

        metrics.update({"run": run, "threads": threads, "streams": streams})

        all_rows.append(metrics)
      except RuntimeError as e:
        print(f"Error during configuration (B:{batch}, T:{threads}, S:{streams}): {e}")
        continue

  fieldnames = [
    "run",
    "threads",
    "streams",
    "batch_size",
    "batch_throughput_batches_per_sec",
    "time_taken_sec",
    "batch_latency_avg_ms",
    "batch_latency_min_ms",
    "batch_latency_max_ms",
    "batch_latency_std_ms",
    "avg_in_flight_measured",
    "avg_in_flight_calculated",
    "max_in_flight",
    "single_throughput_evals_per_sec",
    "single_latency_avg_ms",
  ]

  with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

  print(f"\nAll benchmarks complete. Results written to: {args.output}")


if __name__ == "__main__":
  main()
