import argparse
import csv
import math
import sys
import itertools
from pathlib import Path
from statistics import median

import optuna

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from alphapack import benchmark_inference_engine

DEFAULT_RUN_SIZE = 10_000
DEFAULT_DRY_RUN_SIZE = 64
DEFAULT_BATCH_POOL_SIZE = 512


def run_benchmark(
  model_path: str,
  run_size: int,
  dry_run_size: int,
  batch_size: int,
  batch_pool_size: int,
  threads: int,
  streams: int,
):
  return benchmark_inference_engine(
    model_path=model_path,
    run_size=run_size,
    dry_run_size=dry_run_size,
    batch_size=batch_size,
    batch_pool_size=batch_pool_size,
    thread_pool_size=threads,
    stream_pool_size=streams,
  )


def result_to_metrics(result):
  return {
    "batch_size": result.batch_size,
    "batch_throughput_batches_per_sec": result.batch_throughput_batches_per_sec,
    "time_taken_sec": result.time_taken_sec,
    "batch_latency_avg_ms": result.batch_latency_avg_ms,
    "batch_latency_std_ms": result.batch_latency_std_ms,
    "batch_latency_min_ms": result.batch_latency_min_ms,
    "batch_latency_max_ms": result.batch_latency_max_ms,
    "avg_in_flight_measured": result.avg_in_flight_measured,
    "avg_in_flight_calculated": result.avg_in_flight_calculated,
    "max_in_flight": result.max_in_flight,
    "single_throughput_evals_per_sec": result.single_throughput_evals_per_sec,
    "single_latency_avg_ms": result.single_latency_avg_ms,
  }


def run_grid(args: argparse.Namespace) -> None:
  combinations = list(itertools.product(args.batch_sizes, args.threads, args.streams))
  total_tasks = len(combinations) * args.runs

  print(f"Total Unique Combinations: {len(combinations)}")
  print(f"Total Benchmark Runs: {total_tasks}")

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

  all_rows = []
  for run in range(1, args.runs + 1):
    print(f"\n--- STARTING RUN {run}/{args.runs} ---")

    for batch, threads, streams in combinations:
      try:
        result = run_benchmark(
          model_path=args.model,
          run_size=args.run_size,
          dry_run_size=args.dry_run_size,
          batch_size=batch,
          batch_pool_size=args.batch_pool_size,
          threads=threads,
          streams=streams,
        )
        metrics = result_to_metrics(result)
        metrics.update({"run": run, "threads": threads, "streams": streams})
        all_rows.append(metrics)
      except RuntimeError as e:
        print(f"Error during configuration (B:{batch}, T:{threads}, S:{streams}): {e}")
        continue

  with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

  print(f"\nAll benchmarks complete. Results written to: {args.output}")


def run_optimize(args: argparse.Namespace) -> None:
  fieldnames = [
    "trial_number",
    "batch_size",
    "threads",
    "streams",
    "score",
    "batch_throughput_batches_per_sec",
    "time_taken_sec",
    "batch_latency_avg_ms",
    "batch_latency_std_ms",
    "batch_latency_min_ms",
    "batch_latency_max_ms",
    "avg_in_flight_measured",
    "avg_in_flight_calculated",
    "max_in_flight",
    "single_throughput_evals_per_sec",
    "single_latency_avg_ms",
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
      result = run_benchmark(
        model_path=args.model,
        run_size=args.run_size,
        dry_run_size=args.dry_run_size,
        batch_size=batch,
        batch_pool_size=args.batch_pool_size,
        threads=threads,
        streams=streams,
      )
      all_metrics.append(result_to_metrics(result))

    final_metrics = {key: median([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}

    score = final_metrics["single_throughput_evals_per_sec"]

    row = {"trial_number": trial.number, "threads": threads, "streams": streams, "score": score}
    row.update(final_metrics)
    with open(args.output, "a", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writerow(row)

    return score

  study.optimize(objective, n_trials=args.trials)

  best = study.best_trial
  print("\n--- Optimization Complete ---")
  print(f"Best Score: {best.value:.2f}")
  print(f"Best Params: {best.params}")
  print(f"Full results saved to: {args.output}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Inference Engine benchmark runner.")
  parser.add_argument("--mode", choices=["grid", "optimize"], required=True)
  parser.add_argument("--model", required=True, help="Path to the model file")
  parser.add_argument("--batch_sizes", type=int, nargs="+", required=True)
  parser.add_argument("--threads", type=int, nargs="+", required=True)
  parser.add_argument("--streams", type=int, nargs="+", required=True)
  parser.add_argument("--run_size", type=int, default=DEFAULT_RUN_SIZE)
  parser.add_argument("--dry_run_size", type=int, default=DEFAULT_DRY_RUN_SIZE)
  parser.add_argument("--batch_pool_size", type=int, default=DEFAULT_BATCH_POOL_SIZE)

  parser.add_argument("--runs", type=int, help="Number of times to repeat each combination")
  parser.add_argument("--output", default="benchmark_results.csv")

  parser.add_argument("--trials", type=int, default=20)
  parser.add_argument("--repeats", type=int)
  parser.add_argument("--seed", type=int, default=0)

  args = parser.parse_args()

  if args.mode == "grid":
    if args.runs is None:
      parser.error("--runs is required for grid mode")
    run_grid(args)
  else:
    if args.repeats is None:
      parser.error("--repeats is required for optimize mode")
    run_optimize(args)


if __name__ == "__main__":
  main()
