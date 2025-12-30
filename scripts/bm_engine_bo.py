import argparse
import csv
import math
import re
import subprocess
import optuna
from pathlib import Path
from statistics import median

BATCH_RESULTS_RE = re.compile(
  r"Results \(Batch Size:\s*(\d+)\):\s*"
  r"\n\s*Throughput:\s*([0-9.]+)\s+batches/sec"
  r"\n\s*Time Taken:\s*([0-9.]+)\s+sec"
  r"\n\s*Batch Latency:\s*"
  r"\n\s*Avg:\s*([0-9.]+)\s+ms"
  r"\n\s*Std Dev:\s*([0-9.]+)\s+ms"
  r"\n\s*Min:\s*([0-9.]+)\s+ms"
  r"\n\s*Max:\s*([0-9.]+)\s+ms"
  r"\n\s*In-Flight Batches:\s*"
  r"\n\s*Avg\(Meas\.\):\s*([0-9.]+)"
  r"\n\s*Avg\(Calc\.\):\s*([0-9.]+)"
  r"\n\s*Max:\s*([0-9.]+)",
  re.MULTILINE,
)

SINGLE_RESULTS_RE = re.compile(
  r"Results \(Single Evaluation\):\s*"
  r"\n\s*Throughput:\s*([0-9.]+)\s+evals/sec"
  r"\n\s*Avg Latency:\s*([0-9.]+)\s+ms",
  re.MULTILINE,
)


def run_benchmark(
  exe_path: Path, model_path: Path, batch_size: int, threads: int, streams: int
) -> str:
  cmd = [str(exe_path), str(model_path), str(batch_size), str(threads), str(streams)]
  result = subprocess.run(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
  )
  if result.returncode != 0:
    raise RuntimeError(f"Benchmark failed: {result.stderr}")
  return result.stdout


def parse_output(output: str) -> dict:
  batch_match = BATCH_RESULTS_RE.search(output)
  single_match = SINGLE_RESULTS_RE.search(output)

  if not batch_match or not single_match:
    raise RuntimeError("Missing benchmark results section in output.")

  return {
    "batch_size": int(batch_match.group(1)),
    "batch_throughput_batches_per_sec": float(batch_match.group(2)),
    "time_taken_sec": float(batch_match.group(3)),
    "batch_latency_avg_ms": float(batch_match.group(4)),
    "batch_latency_std_ms": float(batch_match.group(5)),
    "batch_latency_min_ms": float(batch_match.group(6)),
    "batch_latency_max_ms": float(batch_match.group(7)),
    "avg_in_flight_measured": float(batch_match.group(8)),
    "avg_in_flight_calculated": float(batch_match.group(9)),
    "max_in_flight": float(batch_match.group(10)),
    "single_throughput_evals_per_sec": float(single_match.group(1)),
    "single_latency_avg_ms": float(single_match.group(2)),
  }


def calculate_latency_penalty(ms: float) -> float:
  if ms <= 10.0:
    return 1.0
  k = 0.12
  penalty = math.exp(-k * (ms - 10.0))
  return max(0.0, penalty)


def main():
  p = argparse.ArgumentParser(description="Bayesian Optimization for Inference Engine")
  p.add_argument("--exe", required=True)
  p.add_argument("--model", required=True)
  p.add_argument("--batch_sizes", type=int, nargs='+')
  p.add_argument("--threads", type=int, nargs='+')
  p.add_argument("--streams", type=int, nargs='+')
  p.add_argument("--trials", type=int, default=20)
  p.add_argument("--repeats", type=int)
  p.add_argument("--output", default="bo_results.csv")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  fieldnames = [
    "trial_number", "batch_size", "threads", "streams", "score", "batch_throughput_batches_per_sec",
    "time_taken_sec", "batch_latency_avg_ms", "batch_latency_std_ms", "batch_latency_min_ms",
    "batch_latency_max_ms", "avg_in_flight_measured", "avg_in_flight_calculated", "max_in_flight",
    "single_throughput_evals_per_sec", "single_latency_avg_ms"
  ]

  with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

  study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed)
  )

  def objective(trial: optuna.Trial) -> float:
    batch = trial.suggest_categorical("batch_size", args.batch_sizes)
    threads = trial.suggest_categorical("threads", args.threads)
    streams = trial.suggest_categorical("streams", args.streams)

    all_metrics = []
    for _ in range(args.repeats):
      out = run_benchmark(Path(args.exe), Path(args.model), batch, threads, streams)
      all_metrics.append(parse_output(out))

    final_metrics = {key: median([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}

    penalty = calculate_latency_penalty(final_metrics["batch_latency_avg_ms"])
    score = final_metrics["single_throughput_evals_per_sec"] * penalty

    row = {"trial_number": trial.number, "threads": threads, "streams": streams, "score": score}
    row.update(final_metrics)
    with open(args.output, "a", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writerow(row)

    print(
      f"Trial {trial.number}: Batch={batch}, Threads={threads}, Streams={streams}, "
      f"Lat={final_metrics['batch_latency_avg_ms']:.2f}ms, Score={score:.2f}"
    )

    return score

  study.optimize(objective, n_trials=args.trials)

  best = study.best_trial
  print("\n--- Optimization Complete ---")
  print(f"Best Score: {best.value:.2f}")
  print(f"Best Params: {best.params}")
  print(f"Full results saved to: {args.output}")


if __name__ == "__main__":
  main()
