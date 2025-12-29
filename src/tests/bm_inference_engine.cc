#include <core/inference/inference_engine.h>
#include <core/mcts/inference_manager.h>
#include <core/mcts/model_adapter.h>
#include <core/memory/batched_array_pool.h>
#include <core/state/state.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <latch>
#include <numeric>
#include <print>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace alpack;

template <std::size_t N>
using DataPool = BatchedArrayPool<float, N, CudaHostAllocator>;

constexpr std::size_t BATCH_POOL_SIZE = 256;
constexpr std::size_t RUN_SIZE = 10000;

auto parse_argument(const char* arg, const char* name) -> std::size_t {
  try {
    return std::stoull(arg);
  } catch (const std::exception& e) {
    std::println(std::cerr, "Error parsing {} '{}': {}", name, arg, e.what());
  }
  std::exit(1);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::println(std::cerr, "Usage: {} <model_path> [batch_size] [thread_count] [stream_count]", argv[0]);
    return 1;
  }

  std::string model_path = argv[1];
  std::size_t batch_size = argc >= 3 ? parse_argument(argv[2], "batch size") : 96;
  std::size_t thread_count = argc >= 4 ? parse_argument(argv[3], "thread count") : 4;
  std::size_t stream_count = argc >= 5 ? parse_argument(argv[4], "stream count") : 8;

  std::println("--- Inference Engine Benchmark ---");
  std::println("Model: {}", model_path);
  std::println("Batch Size: {}", batch_size);
  std::println("Threads: {}", thread_count);
  std::println("Streams: {}", stream_count);
  std::println("Total Evaluations: {}", RUN_SIZE);

  try {
    DataPool<ModelAdapter::image_input_size> image_input_storage{batch_size, BATCH_POOL_SIZE};
    DataPool<ModelAdapter::additional_input_size> additional_input_storage{batch_size, BATCH_POOL_SIZE};
    DataPool<ModelAdapter::priors_output_size> priors_output_storage{batch_size, BATCH_POOL_SIZE};
    DataPool<ModelAdapter::value_output_size> value_output_storage{batch_size, BATCH_POOL_SIZE};

    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    std::ranges::generate(image_input_storage.pool(), [&] { return dist(gen); });
    std::ranges::generate(additional_input_storage.pool(), [&] { return dist(gen); });

    std::ifstream file{model_path, std::ios::binary};
    if (!file) {
      throw std::runtime_error("Failed to open model file: " + model_path);
    }

    InferenceModel model{file};
    InferenceEngine engine{std::move(model), stream_count};

    std::latch start_latch{1};

    constexpr auto cache_line_size = std::hardware_destructive_interference_size;
    alignas(cache_line_size) std::atomic<std::size_t> global_idx{0};
    alignas(cache_line_size) std::atomic<std::size_t> finished_cnt{0};
    alignas(cache_line_size) std::atomic<std::int64_t> in_flight_cnt{0};
    alignas(cache_line_size) [[maybe_unused]] std::byte padding{};

    std::vector<double> latencies(RUN_SIZE);

    struct Callback {
      std::atomic<std::size_t>* finished_cnt;
      std::atomic<std::int64_t>* in_flight_cnt;
      double* latency;
      std::chrono::steady_clock::time_point begin;
      InferenceCallback callback;
    };

    std::vector<Callback> callbacks(RUN_SIZE);
    for (auto&& [callback, latency] : std::views::zip(callbacks, latencies)) {
      callback.finished_cnt = &finished_cnt;
      callback.in_flight_cnt = &in_flight_cnt;
      callback.latency = &latency;
      callback.callback.func = [](void* data) {
        const auto end = std::chrono::steady_clock::now();
        const auto& cb = *static_cast<Callback*>(data);
        *cb.latency = std::chrono::duration<double, std::milli>(end - cb.begin).count();
        cb.finished_cnt->fetch_add(1, std::memory_order_relaxed);
        cb.in_flight_cnt->fetch_sub(1, std::memory_order_relaxed);
      };
      callback.callback.data = &callback;
    }

    auto worker_task = [&] {
      start_latch.wait();

      while (true) {
        const auto idx = global_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= RUN_SIZE) {
          break;
        }

        const auto batch_idx = idx % BATCH_POOL_SIZE;
        InferenceInfo info{
          .image_input = image_input_storage.batch(batch_idx),
          .image_input_shape =
            {static_cast<std::int64_t>(batch_size),
             ModelAdapter::input_feature_count,
             State::bin_length,
             State::bin_length},
          .additional_input = additional_input_storage.batch(batch_idx),
          .additional_input_shape = {static_cast<std::int64_t>(batch_size), ModelAdapter::additional_input_count},
          .policy_output = priors_output_storage.batch(batch_idx),
          .policy_output_shape = {static_cast<std::int64_t>(batch_size), State::bin_base_size},
          .value_output = value_output_storage.batch(batch_idx),
          .value_output_shape = {static_cast<std::int64_t>(batch_size), ModelAdapter::value_support_count},
        };

        callbacks[idx].begin = std::chrono::steady_clock::now();
        in_flight_cnt.fetch_add(1, std::memory_order_relaxed);
        engine.run(info, callbacks[idx].callback);
      }
    };

    std::vector<std::jthread> threads;
    threads.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) {
      threads.emplace_back(worker_task);
    }

    std::println("Starting...");
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    start_latch.count_down();

    std::int64_t in_flight_sample_count{}, in_flight_sum{}, in_flight_max{};
    while (true) {
      std::size_t current = finished_cnt.load(std::memory_order_relaxed);
      std::int64_t in_flight = in_flight_cnt.load(std::memory_order_relaxed);
      ++in_flight_sample_count;
      in_flight_sum += in_flight;
      in_flight_max = std::max(in_flight_max, in_flight);

      double percentage = (static_cast<double>(current) * 100.0) / RUN_SIZE;
      std::print("Progress: {:6.2f}%\r", percentage);

      if (current >= RUN_SIZE) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::ranges::for_each(threads, &std::jthread::join);
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    std::println("\nFinished.");

    std::chrono::duration<double> total_time_sec = benchmark_end - benchmark_start;
    double throughput = static_cast<double>(RUN_SIZE) / total_time_sec.count();

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();
    double sq_sum = std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0);
    double std_dev = std::sqrt(std::max(0.0, sq_sum / latencies.size() - mean * mean));
    auto [min_it, max_it] = std::minmax_element(latencies.begin(), latencies.end());

    double in_flight_mean = static_cast<double>(in_flight_sum) / in_flight_sample_count;

    double calculated_in_flight = throughput * (mean / 1000.0);

    std::println("--------------------------------");
    std::println("Results (Batch Size: {}):", batch_size);
    std::println("  Throughput:      {:.2f} batches/sec", throughput);
    std::println("  Time Taken:      {:.2f} sec", total_time_sec.count());
    std::println("  Batch Latency:");
    std::println("    Avg:           {:.2f} ms", mean);
    std::println("    Min:           {:.2f} ms", *min_it);
    std::println("    Max:           {:.2f} ms", *max_it);
    std::println("    Std Dev:       {:.2f} ms", std_dev);
    std::println("  In-Flight Batches:");
    std::println("    Avg(Meas.):    {:.2f}", in_flight_mean);
    std::println("    Avg(Calc.):    {:.2f}", calculated_in_flight);
    std::println("    Max:           {}", in_flight_max);

    std::println("--------------------------------");
    std::println("Results (Single Evaluation):");
    std::println("  Throughput:      {:.2f} evals/sec", throughput * batch_size);
    std::println("  Latency:");
    std::println("    Avg:           {:.4f} ms", mean / batch_size);
    std::println("--------------------------------");

  } catch (const std::exception& e) {
    std::println(std::cerr, "Error: {}", e.what());
    return 1;
  }

  return 0;
}