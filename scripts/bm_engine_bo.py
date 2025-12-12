import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

import optuna

BATCH_SIZES = list(range(16, 209, 16))
THREAD_CHOICES = [1] + list(range(2, 25, 2))

W_TPUT = 0.80
W_SLAT = 0.20

SINGLE_RESULTS_RE = re.compile(
  r"Results \(Single Evaluation\):\s*"
  r"\n\s*Throughput:\s*([0-9.]+)\s+evals/sec"
  r"\n\s*Avg In-Flight:\s*([0-9.]+)\s+evals"
  r"\n\s*Latency:\s*"
  r"\n\s*Avg:\s*([0-9.]+)\s+ms",
  re.MULTILINE,
)


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


def parse_output(output: str) -> dict:
  single_match = SINGLE_RESULTS_RE.search(output)
  if not single_match:
    raise RuntimeError("Missing single-evaluation results section.")

  return {
    "single_throughput_evals_per_sec": float(single_match.group(1)),
    "single_latency_avg_ms": float(single_match.group(3)),
  }


def eval_config(
  exe_path: Path, model_path: Path, batch_size: int, threads: int, streams: int, repeats: int
) -> dict:
  single_tputs = []
  single_lats = []
  for _ in range(repeats):
    if _ == 0: print()
    out = run_benchmark(exe_path, model_path, batch_size, threads, streams)
    m = parse_output(out)
    single_tputs.append(m["single_throughput_evals_per_sec"])
    single_lats.append(m["single_latency_avg_ms"])
  return {
    "single_throughput_evals_per_sec": median(single_tputs),
    "single_latency_avg_ms": median(single_lats),
  }


def norm_throughput(tput: float, best: float = 70000.0) -> float:
  if tput <= 0:
    return 0.0
  return max(0.0, min(1.0, math.log1p(tput) / math.log1p(best)))


def norm_single_latency(ms: float, worst: float = 1.0) -> float:
  return max(0.0, min(1.0, 1.0 - ms / worst))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--exe", required=True)
  p.add_argument("--model", required=True)
  p.add_argument("--trials", type=int, default=20)
  p.add_argument("--repeats", type=int, default=5)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--study", default="benchmark_bo")
  p.add_argument("--storage", default=None)
  p.add_argument("--norm", default=None)
  args = p.parse_args()

  exe_path = Path(args.exe)
  model_path = Path(args.model)

  if args.norm:
    cfg = json.loads(args.norm)
    tput_best = float(cfg.get("single_throughput_best_evals_per_sec", 70000.0))
    single_worst = float(cfg.get("single_latency_worst_ms", 1.0))
  else:
    tput_best = 70000.0
    single_worst = 1.0

  sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
  pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

  study = optuna.create_study(
    study_name=args.study,
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
    storage=args.storage,
    load_if_exists=bool(args.storage),
  )

  def objective(trial: optuna.Trial) -> float:
    batch = trial.suggest_categorical("batch_size", BATCH_SIZES)
    threads = trial.suggest_categorical("threads", THREAD_CHOICES)
    streams = trial.suggest_int("streams", 1, 16)

    m = eval_config(exe_path, model_path, batch, threads, streams, repeats=args.repeats)

    nt = 0.0 if m["single_throughput_evals_per_sec"] <= 0 else max(
      0.0, min(1.0,
               math.log1p(m["single_throughput_evals_per_sec"]) / math.log1p(tput_best))
    )

    ns = max(0.0, min(1.0, 1.0 - m["single_latency_avg_ms"] / single_worst))

    eps = 1e-9
    denominator = (W_TPUT / max(nt, eps)) + (W_SLAT / max(ns, eps))
    score = 1.0 / denominator

    trial.set_user_attr("metrics", m)
    trial.set_user_attr("norm_throughput", nt)
    trial.set_user_attr("norm_single_latency", ns)
    trial.set_user_attr("score", score)

    print(
      f"Trial {trial.number}: batch={batch} threads={threads} streams={streams} score={score:.6f}"
    )
    print(f"metrics={m} norms={{'tput': {nt:.6f}, 'single': {ns:.6f}}}")

    return score

  study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

  best = study.best_trial
  print("params:", best.params)
  print("score:", best.value)
  print("metrics:", best.user_attrs.get("metrics"))
  print(
    "norms:", {
      "throughput": best.user_attrs.get("norm_throughput"),
      "single_latency": best.user_attrs.get("norm_single_latency"),
    }
  )


if __name__ == "__main__":
  main()
