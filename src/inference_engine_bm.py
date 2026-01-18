import argparse
import csv
import itertools

from tqdm import tqdm
from alphapack import benchmark_inference_engine


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


def run(args: argparse.Namespace) -> None:
  combinations = list(itertools.product(args.batch_sizes, args.threads, args.streams))
  total_tasks = len(combinations) * args.runs

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
    with tqdm(total=total_tasks) as progress:
      for run in range(args.runs):
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
            writer.writerow(metrics)
            f.flush()
          except RuntimeError as e:
            print(f"(B:{batch}, T:{threads}, S:{streams})\n{e}")
            continue
          finally:
            progress.update(1)


def main() -> None:
  DEFAULT_RUN_SIZE = 10_000
  DEFAULT_DRY_RUN_SIZE = 64
  DEFAULT_BATCH_POOL_SIZE = 512

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True)
  parser.add_argument("--batch_sizes", type=int, nargs="+", required=True)
  parser.add_argument("--threads", type=int, nargs="+", required=True)
  parser.add_argument("--streams", type=int, nargs="+", required=True)
  parser.add_argument("--run_size", type=int, default=DEFAULT_RUN_SIZE)
  parser.add_argument("--dry_run_size", type=int, default=DEFAULT_DRY_RUN_SIZE)
  parser.add_argument("--batch_pool_size", type=int, default=DEFAULT_BATCH_POOL_SIZE)
  parser.add_argument("--runs", type=int, required=True)
  parser.add_argument("--output", default="results.csv")

  args = parser.parse_args()
  run(args)


if __name__ == "__main__":
  main()
