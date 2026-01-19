#pragma once
#include <string>

namespace alpack {

struct InferenceEngineBenchmarkInfo {
  std::string model_path;
  std::size_t run_size;
  std::size_t dry_run_size;
  std::size_t batch_size;
  std::size_t batch_pool_size;
  std::size_t thread_pool_size;
  std::size_t stream_pool_size;
};

struct InferenceEngineBenchmarkResult {
  std::string model_path;
  std::size_t run_size;
  std::size_t batch_size;
  std::size_t thread_pool_size;
  std::size_t stream_pool_size;
  double batch_throughput_batches_per_sec;
  double time_taken_sec;
  double batch_latency_avg_ms;
  double batch_latency_std_ms;
  double batch_latency_min_ms;
  double batch_latency_max_ms;
  double avg_in_flight_measured;
  double avg_in_flight_calculated;
  std::size_t max_in_flight;
  double single_throughput_evals_per_sec;
  double single_latency_avg_ms;
};

auto benchmark_inference_engine(const InferenceEngineBenchmarkInfo& info)
  -> InferenceEngineBenchmarkResult;

}  // namespace alpack
