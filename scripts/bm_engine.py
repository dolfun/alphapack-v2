import argparse
import csv
import subprocess
import sys
import re
from pathlib import Path

BATCH_SIZES = [16, 32, 64, 96, 128, 192, 256]
NUM_RUNS = 4

BATCH_RESULTS_RE = re.compile(
  r"Results \(Batch Size:\s*(\d+)\):\s*"
  r"\n\s*Throughput:\s*([0-9.]+)\s+batches/sec"
  r"\n\s*Time Taken:\s*([0-9.]+)\s+sec"
  r"\n\s*Avg In-Flight:\s*([0-9.]+)\s+batches"
  r"\n\s*Batch Latency:\s*"
  r"\n\s*Avg:\s*([0-9.]+)\s+ms"
  r"\n\s*Min:\s*([0-9.]+)\s+ms"
  r"\n\s*Max:\s*([0-9.]+)\s+ms"
  r"\n\s*Std Dev:\s*([0-9.]+)\s+ms", re.MULTILINE
)

SINGLE_RESULTS_RE = re.compile(
  r"Results \(Single Evaluation\):\s*"
  r"\n\s*Throughput:\s*([0-9.]+)\s+evals/sec"
  r"\n\s*Avg In-Flight:\s*([0-9.]+)\s+evals"
  r"\n\s*Latency:\s*"
  r"\n\s*Avg:\s*([0-9.]+)\s+ms", re.MULTILINE
)


def parse_output(output: str, expected_batch_size: int):
  batch_match = BATCH_RESULTS_RE.search(output)
  if not batch_match:
    raise RuntimeError("Missing batch-size results section.")

  single_match = SINGLE_RESULTS_RE.search(output)
  if not single_match:
    raise RuntimeError("Missing single-evaluation results section.")

  batch_size = int(batch_match.group(1))
  if batch_size != expected_batch_size:
    raise RuntimeError(f"Expected batch {expected_batch_size}, got {batch_size}")

  return {
    "batch_size": batch_size,
    "batch_throughput_batches_per_sec": float(batch_match.group(2)),
    "time_taken_sec": float(batch_match.group(3)),
    "avg_in_flight_batches": float(batch_match.group(4)),
    "batch_latency_avg_ms": float(batch_match.group(5)),
    "batch_latency_min_ms": float(batch_match.group(6)),
    "batch_latency_max_ms": float(batch_match.group(7)),
    "batch_latency_std_ms": float(batch_match.group(8)),
    "single_throughput_evals_per_sec": float(single_match.group(1)),
    "avg_in_flight_evals": float(single_match.group(2)),
    "single_latency_avg_ms": float(single_match.group(3)),
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
    raise RuntimeError("Benchmark failed.")

  return result.stdout


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--exe", required=True, help="Path to the C++ executable")
  parser.add_argument("--model", required=True, help="Path to the model file")
  parser.add_argument("--threads", type=int, default=16, help="Number of threads (default: 16)")
  parser.add_argument("--streams", type=int, default=16, help="Number of streams (default: 16)")
  parser.add_argument("--output", default="benchmark_results.csv", help="Output CSV file path")
  args = parser.parse_args()

  exe_path = Path(args.exe)
  model_path = Path(args.model)

  all_rows = []

  for run in range(1, NUM_RUNS + 1):
    print(f"\n=== RUN {run} ===")

    for batch in BATCH_SIZES:
      output = run_benchmark(exe_path, model_path, batch, args.threads, args.streams)
      metrics = parse_output(output, batch)

      metrics["run"] = run
      metrics["threads"] = args.threads
      metrics["streams"] = args.streams

      all_rows.append(metrics)

  fieldnames = [
    "run",
    "threads",
    "streams",
    "batch_size",
    "batch_throughput_batches_per_sec",
    "time_taken_sec",
    "avg_in_flight_batches",
    "batch_latency_avg_ms",
    "batch_latency_min_ms",
    "batch_latency_max_ms",
    "batch_latency_std_ms",
    "single_throughput_evals_per_sec",
    "avg_in_flight_evals",
    "single_latency_avg_ms",
  ]

  with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

  print(f"\n📄 Results written to: {args.output}")


if __name__ == "__main__":
  main()
