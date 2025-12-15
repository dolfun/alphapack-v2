#include <core/mcts/model_info.h>
#include <core/state/state.h>
#include <core/torch_utils/inference_engine.h>
#include <core/torch_utils/memory.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <forward_list>
#include <fstream>
#include <iostream>
#include <latch>
#include <memory_resource>
#include <numeric>
#include <print>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace alpack;

constexpr size_t batch_count = 128;
constexpr size_t run_size = 10000;

auto read_file(const std::string& path) -> std::string {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open model file: " + path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

auto fill_random(float* ptr, size_t count) -> void {
  static std::mt19937 gen{std::random_device{}()};
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < count; ++i) {
    ptr[i] = dist(gen);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::println(
      std::cerr,
      "Usage: {} <model_path> [batch_size] [thread_count] [stream_count]",
      argv[0]
    );
    return 1;
  }

  std::string model_path = argv[1];
  int64_t batch_size = 32;
  size_t thread_count = 16;
  size_t stream_count = 4;

  if (argc >= 3) {
    try {
      batch_size = std::stoll(argv[2]);
      if (batch_size <= 0) {
        throw std::invalid_argument("must be > 0");
      }
    } catch (const std::exception& e) {
      std::println(std::cerr, "Error parsing batch size '{}': {}", argv[2], e.what());
      return 1;
    }
  }

  if (argc >= 4) {
    try {
      thread_count = std::stoll(argv[3]);
      if (thread_count <= 0) {
        throw std::invalid_argument("must be > 0");
      }
    } catch (const std::exception& e) {
      std::println(std::cerr, "Error parsing thread count '{}': {}", argv[3], e.what());
      return 1;
    }
  }

  if (argc >= 5) {
    try {
      stream_count = std::stoll(argv[4]);
      if (stream_count <= 0) {
        throw std::invalid_argument("must be > 0");
      }
    } catch (const std::exception& e) {
      std::println(std::cerr, "Error parsing stream count '{}': {}", argv[4], e.what());
      return 1;
    }
  }

  constexpr size_t estimated_node_size = sizeof(size_t) + sizeof(void*) + alignof(std::max_align_t);
  constexpr size_t buffer_size = run_size * estimated_node_size;
  alignas(std::max_align_t) std::byte buffer[buffer_size];
  std::pmr::monotonic_buffer_resource pool(buffer, buffer_size, std::pmr::null_memory_resource());

  std::println("--- Inference Engine Benchmark ---");
  std::println("Model: {}", model_path);
  std::println("Batch Size: {}", batch_size);
  std::println("Threads: {}", thread_count);
  std::println("Streams: {}", stream_count);
  std::println("Total Evaluations: {}", run_size);

  constexpr size_t bin_size = State::bin_length * State::bin_length;

  const size_t image_elements_per_batch = batch_size * ModelInfo::input_feature_count * bin_size;
  const size_t image_bytes_per_batch = image_elements_per_batch * sizeof(float);

  const size_t additional_elements_per_batch = batch_size * ModelInfo::additional_input_count;
  const size_t additional_bytes_per_batch = additional_elements_per_batch * sizeof(float);

  const size_t policy_elements_per_batch = batch_size * bin_size;
  const size_t policy_bytes_per_batch = policy_elements_per_batch * sizeof(float);

  const size_t value_elements_per_batch = batch_size * ModelInfo::value_support_count;
  const size_t value_bytes_per_batch = value_elements_per_batch * sizeof(float);

  try {
    using PinnedPool = PinnedMemoryPool<alignof(std::max_align_t)>;
    PinnedPool image_input_pool{image_bytes_per_batch, batch_count};
    PinnedPool additional_input_pool{additional_bytes_per_batch, batch_count};
    PinnedPool policy_output_pool{policy_bytes_per_batch, batch_count};
    PinnedPool value_output_pool{value_bytes_per_batch, batch_count};

    for (size_t i = 0; i < static_cast<size_t>(batch_count); ++i) {
      fill_random(static_cast<float*>(image_input_pool[i]), image_elements_per_batch);
      fill_random(static_cast<float*>(additional_input_pool[i]), additional_elements_per_batch);
    }

    std::ifstream file{model_path, std::ios::binary};
    if (!file) {
      throw std::runtime_error("Failed to open model file: " + model_path);
    }

    InferenceModel model{file};
    InferenceEngine engine{std::move(model), stream_count};

    std::latch start_latch{1};
    std::atomic<int64_t> in_flight_counter{0};
    std::atomic<size_t> global_idx{0};
    std::vector<std::pair<InferenceResult, std::chrono::high_resolution_clock::time_point>> results(
      run_size
    );

    auto worker_task = [&] {
      start_latch.wait();

      while (true) {
        size_t idx = global_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= run_size) {
          break;
        }

        size_t slot = idx % batch_count;

        InferenceInfo info{
          .image_input_shape =
            {batch_size, ModelInfo::input_feature_count, State::bin_length, State::bin_length},
          .image_input_ptr = image_input_pool[slot],

          .additional_input_shape = {batch_size, ModelInfo::additional_input_count},
          .additional_input_ptr = additional_input_pool[slot],

          .policy_output_shape = {batch_size, bin_size},
          .policy_output_ptr = policy_output_pool[slot],

          .value_output_shape = {batch_size, ModelInfo::value_support_count},
          .value_output_ptr = value_output_pool[slot]
        };

        auto start_time = std::chrono::high_resolution_clock::now();
        in_flight_counter.fetch_add(1, std::memory_order_relaxed);
        auto result = engine.run(info);
        results[idx] = {std::move(result), start_time};
      }
    };

    std::vector<std::jthread> threads;
    threads.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
      threads.emplace_back(worker_task);
    }

    std::pmr::forward_list<size_t> active_indices(&pool);
    for (size_t i = run_size; i > 0; --i) {
      active_indices.push_front(i - 1);
    }

    int64_t in_flight_counter_sum = 0;
    size_t in_flight_counter_updates = 0;
    std::vector<double> latencies(run_size);

    std::println("Starting...");
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    start_latch.count_down();

    while (!active_indices.empty()) {
      auto prev = active_indices.before_begin();
      auto it = active_indices.begin();

      while (it != active_indices.end()) {
        size_t idx = *it;
        auto& [res, start_time] = results[idx];

        if (start_time.time_since_epoch().count() == 0) {
          break;
        }

        if (res.is_done()) {
          in_flight_counter.fetch_sub(1, std::memory_order_relaxed);
          auto end_time = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double, std::milli> latency_ms = end_time - start_time;
          latencies[idx] = latency_ms.count();
          it = active_indices.erase_after(prev);
        } else {
          prev = it;
          ++it;
        }
      }

      in_flight_counter_sum += in_flight_counter.load(std::memory_order_relaxed);
      ++in_flight_counter_updates;
    }

    threads.clear();
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    std::println("Finished.");

    std::chrono::duration<double> total_time_sec = benchmark_end - benchmark_start;
    double throughput = static_cast<double>(run_size) / total_time_sec.count();

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();

    double sq_sum = std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / latencies.size() - mean * mean);

    auto [min_it, max_it] = std::minmax_element(latencies.begin(), latencies.end());
    double min_latency = *min_it;
    double max_latency = *max_it;

    double avg_in_flight_batches = 0.0;
    if (in_flight_counter_updates > 0) {
      avg_in_flight_batches =
        static_cast<double>(in_flight_counter_sum) / in_flight_counter_updates;
    }

    std::println("--------------------------------");
    std::println("Results (Batch Size: {}):", batch_size);
    std::println("  Throughput:    {:.2f} batches/sec", throughput);
    std::println("  Time Taken:    {:.2f} sec", total_time_sec.count());
    std::println("  Avg In-Flight: {:.2f} batches", avg_in_flight_batches);
    std::println("  Batch Latency:");
    std::println("    Avg:         {:.2f} ms", mean);
    std::println("    Min:         {:.2f} ms", min_latency);
    std::println("    Max:         {:.2f} ms", max_latency);
    std::println("    Std Dev:     {:.2f} ms", std_dev);

    std::println("--------------------------------");
    std::println("Results (Single Evaluation):");
    std::println("  Throughput:    {:.2f} evals/sec", throughput * batch_size);
    std::println("  Avg In-Flight: {:.2f} evals", avg_in_flight_batches * batch_size);
    std::println("  Latency:");
    std::println("    Avg:         {:.4f} ms", mean / batch_size);
    std::println("--------------------------------");

  } catch (const std::exception& e) {
    std::println(std::cerr, "Error: {}", e.what());
    return -1;
  }

  return 0;
}