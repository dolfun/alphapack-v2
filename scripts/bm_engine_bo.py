import argparse
import math
import re
import subprocess
from pathlib import Path
from statistics import median
import optuna

BATCH_SIZES = [64, 96, 128, 192]
THREAD_CHOICES = list(range(2, 21, 2))
STREAM_CHOICES = list(range(2, 16, 2))

BATCH_LATENCY_RE = re.compile(
  r"Results \(Batch Size:.*?\):.*?Batch Latency:\s*\n\s*Avg:\s*([0-9.]+)\s*ms", re.DOTALL
)

SINGLE_TPUT_RE = re.compile(
  r"Results \(Single Evaluation\):\s*\n\s*Throughput:\s*([0-9.]+)\s*evals/sec", re.DOTALL
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
  batch_match = BATCH_LATENCY_RE.search(output)
  single_match = SINGLE_TPUT_RE.search(output)

  if not batch_match or not single_match:
    # Debugging: print output if parsing fails to see why
    # print(output)
    raise RuntimeError("Missing benchmark results section in output.")

  return {
    "single_throughput_evals_per_sec": float(single_match.group(1)),
    "batch_latency_avg_ms": float(batch_match.group(1)),
  }


def eval_config(
  exe_path: Path, model_path: Path, batch_size: int, threads: int, streams: int, repeats: int
) -> dict:
  tputs, lats = [], []
  for _ in range(repeats):
    out = run_benchmark(exe_path, model_path, batch_size, threads, streams)
    m = parse_output(out)
    tputs.append(m["single_throughput_evals_per_sec"])
    lats.append(m["batch_latency_avg_ms"])
  return {
    "single_throughput_evals_per_sec": median(tputs),
    "batch_latency_avg_ms": median(lats),
  }


def calculate_latency_penalty(ms: float) -> float:
  if ms <= 10.0:
    return 1.0
  k = 0.12
  penalty = math.exp(-k * (ms - 10.0))
  return max(0.0, penalty)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--exe", required=True)
  p.add_argument("--model", required=True)
  p.add_argument("--trials", type=int, default=20)
  p.add_argument("--repeats", type=int, default=5)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed)
  )

  def objective(trial: optuna.Trial) -> float:
    batch = trial.suggest_categorical("batch_size", BATCH_SIZES)
    threads = trial.suggest_categorical("threads", THREAD_CHOICES)
    streams = trial.suggest_categorical("streams", STREAM_CHOICES)

    m = eval_config(Path(args.exe), Path(args.model), batch, threads, streams, repeats=args.repeats)
    tput = m["single_throughput_evals_per_sec"]
    lat = m["batch_latency_avg_ms"]

    penalty = calculate_latency_penalty(lat)
    score = tput * penalty

    trial.set_user_attr("batch_latency", lat)
    trial.set_user_attr("throughput", tput)

    print(
      f"Trial {trial.number}: Batch={batch}, Latency={lat:.2f}ms, Tput={tput:.2f}, Score={score:.2f}"
    )
    return score

  study.optimize(objective, n_trials=args.trials)

  best = study.best_trial
  print("\nBest Config:")
  print(f"Params: {best.params}")
  print(f"Throughput: {best.user_attrs['throughput']}")
  print(f"Latency: {best.user_attrs['batch_latency']}")


if __name__ == "__main__":
  main()
