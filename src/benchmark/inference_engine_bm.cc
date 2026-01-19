#include "inference_engine_bm.h"

#include <core/inference/allocator.h>
#include <core/inference/inference_engine.h>
#include <core/mcts/model_adapter.h>
#include <core/memory/batched_array_pool.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <latch>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <ranges>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace alpack {

namespace {

struct DataPool {
  DataPool(std::size_t batch_size, std::size_t batch_pool_size)
      : image_input{batch_size, batch_pool_size},
        additional_input{batch_size, batch_pool_size},
        priors_output{batch_size, batch_pool_size},
        value_output{batch_size, batch_pool_size} {
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    std::ranges::generate(image_input.pool(), [&] { return dist(gen); });
    std::ranges::generate(additional_input.pool(), [&] { return dist(gen); });
  }

  template <std::size_t N>
  using HostPool = BatchedArrayPool<float, N, CudaHostAllocator>;

  HostPool<ModelAdapter::image_input_size> image_input;
  HostPool<ModelAdapter::additional_input_size> additional_input;
  HostPool<ModelAdapter::priors_output_size> priors_output;
  HostPool<ModelAdapter::value_output_size> value_output;
};

template <typename T>
class CircularQueue {
public:
  explicit CircularQueue(std::size_t capacity)
      : m_capacity{capacity}, m_head{0}, m_tail{0}, m_buffer(m_capacity), m_size{0} {}

  auto push(const T& val) -> void {
    if (m_tail - m_head >= m_capacity) {
      throw std::runtime_error("Queue overflow");
    }

    m_buffer[m_tail % m_capacity] = val;
    ++m_tail;

    m_size.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] auto pop() -> T {
    if (m_head == m_tail) {
      throw std::runtime_error("Queue underflow");
    }

    auto val = m_buffer[m_head % m_capacity];
    ++m_head;
    m_size.fetch_sub(1, std::memory_order_relaxed);

    return val;
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t {
    return m_size.load(std::memory_order_relaxed);
  }

private:
  std::size_t m_capacity, m_head, m_tail;
  std::vector<T> m_buffer;
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> m_size;
  alignas(std::hardware_destructive_interference_size) std::byte padding{};
};

template <typename T>
  requires std::is_default_constructible_v<T>
class ObjectPool {
public:
  explicit ObjectPool(std::size_t size) : m_pool{size}, m_queue{size} {
    std::ranges::for_each(m_pool, [this](T& obj) { m_queue.push(std::addressof(obj)); });
  }

  auto allocate() -> T* {
    std::lock_guard _{m_mutex};
    return m_queue.pop();
  }

  auto free(T* ptr) -> void {
    std::lock_guard _{m_mutex};
    m_queue.push(ptr);
  }

  [[nodiscard]] auto in_flight_count() const noexcept -> std::size_t {
    return m_pool.size() - m_queue.size();
  }

private:
  std::vector<T> m_pool;
  CircularQueue<T*> m_queue;
  std::mutex m_mutex;
};

class InFlightCounter {
public:
  auto update(std::size_t val) noexcept -> void {
    ++sample_count;
    sum += val;
    max = std::max(max, val);
  }

  [[nodiscard]] auto stats() const noexcept -> std::pair<double, std::size_t> {
    if (sample_count == 0) {
      return {0.0, 0};
    }
    auto mean = static_cast<double>(sum) / static_cast<double>(sample_count);
    return {mean, max};
  }

private:
  std::size_t sample_count{0}, sum{0}, max{0};
};

class BenchmarkState {
public:
  BenchmarkState(InferenceEngineBenchmarkInfo, InferenceModel);

  auto run() -> void;
  [[nodiscard]] auto results() const -> InferenceEngineBenchmarkResult;

private:
  auto warmup() -> void;
  auto task() -> void;
  auto get_inference_info(std::size_t idx) -> InferenceInfo;

  static auto callback_func(void* data) -> void {
    auto& run_info = *static_cast<RunInfo*>(data);
    run_info.t1 = std::chrono::steady_clock::now();
    run_info.finish_cnt->fetch_add(1, std::memory_order_relaxed);
    run_info.callback_pool->free(run_info.callback);
  }

  InferenceEngineBenchmarkInfo m_info;
  InferenceEngine m_engine;
  DataPool m_storage;
  ObjectPool<InferenceCallback> m_callback_pool;

  struct RunInfo {
    ObjectPool<InferenceCallback>* callback_pool{};
    InferenceCallback* callback{};
    std::atomic<std::size_t>* finish_cnt{};
    std::chrono::steady_clock::time_point t0, t1;
  };
  std::vector<RunInfo> m_run_infos;

  std::latch latch;
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> m_start_cnt;
  alignas(std::hardware_destructive_interference_size) std::atomic<std::size_t> m_finish_cnt;
  alignas(std::hardware_destructive_interference_size) [[maybe_unused]] std::byte padding{};

  InFlightCounter m_in_flight_counter;
  std::chrono::duration<double> m_time_taken{};
};

}  // namespace

BenchmarkState::BenchmarkState(InferenceEngineBenchmarkInfo info, InferenceModel model)
    : m_info{std::move(info)},
      m_engine{std::move(model), m_info.stream_pool_size},
      m_storage{m_info.batch_size, m_info.batch_pool_size},
      m_callback_pool{m_info.batch_pool_size},
      m_run_infos(m_info.run_size),
      latch{1},
      m_start_cnt{0},
      m_finish_cnt{0} {
  std::ranges::for_each(m_run_infos, [this](auto& run_info) {
    run_info.callback_pool = &m_callback_pool;
    run_info.finish_cnt = &m_finish_cnt;
  });
}

auto BenchmarkState::run() -> void {
  warmup();

  auto threads =
    std::views::iota(0uz, m_info.thread_pool_size) |
    std::views::transform([this](auto) { return std::jthread{&BenchmarkState::task, this}; }) |
    std::ranges::to<std::vector>();

  const auto benchmark_start = std::chrono::steady_clock::now();
  latch.count_down();
  while (true) {
    const auto progress = m_finish_cnt.load(std::memory_order_relaxed);
    if (progress >= m_info.run_size) break;

    m_in_flight_counter.update(m_callback_pool.in_flight_count());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::ranges::for_each(threads, &std::jthread::join);
  const auto benchmark_end = std::chrono::steady_clock::now();
  m_time_taken = benchmark_end - benchmark_start;
}

auto BenchmarkState::results() const -> InferenceEngineBenchmarkResult {
  const double elapsed = m_time_taken.count();
  const double throughput =
    (elapsed > 0.0 && m_info.run_size > 0) ? static_cast<double>(m_info.run_size) / elapsed : 0.0;

  double avg_lat = 0.0;
  double std_lat = 0.0;
  double min_lat = 0.0;
  double max_lat = 0.0;
  double calculated_in_flight = 0.0;
  if (!m_run_infos.empty()) {
    const auto latencies =
      m_run_infos | std::views::transform([](const RunInfo& cb) {
        return std::chrono::duration<double, std::milli>(cb.t1 - cb.t0).count();
      }) |
      std::ranges::to<std::vector>();
    const auto size = static_cast<double>(latencies.size());
    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double mean = sum / size;
    const double sq_sum =
      std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0);
    const double std_dev = std::sqrt(std::max(0.0, sq_sum / size - mean * mean));
    const auto [min_it, max_it] = std::minmax_element(latencies.begin(), latencies.end());
    avg_lat = mean;
    std_lat = std_dev;
    min_lat = *min_it;
    max_lat = *max_it;
    calculated_in_flight = throughput * (mean / 1000.0);
  }

  const auto [in_flight_mean, in_flight_max] = m_in_flight_counter.stats();

  return {
    .model_path = m_info.model_path,
    .run_size = m_info.run_size,
    .batch_size = m_info.batch_size,
    .thread_pool_size = m_info.thread_pool_size,
    .stream_pool_size = m_info.stream_pool_size,
    .batch_throughput_batches_per_sec = throughput,
    .time_taken_sec = elapsed,
    .batch_latency_avg_ms = avg_lat,
    .batch_latency_std_ms = std_lat,
    .batch_latency_min_ms = min_lat,
    .batch_latency_max_ms = max_lat,
    .avg_in_flight_measured = in_flight_mean,
    .avg_in_flight_calculated = calculated_in_flight,
    .max_in_flight = in_flight_max,
    .single_throughput_evals_per_sec = throughput * static_cast<double>(m_info.batch_size),
    .single_latency_avg_ms =
      (m_info.batch_size > 0) ? avg_lat / static_cast<double>(m_info.batch_size) : 0.0
  };
}

auto BenchmarkState::warmup() -> void {
  std::binary_semaphore semaphore{0};
  InferenceCallback cb;
  cb.data = &semaphore;
  cb.func = [](void* data) {
    auto& _semaphore = *static_cast<std::binary_semaphore*>(data);
    _semaphore.release();
  };

  const auto inference_info = get_inference_info(0);
  for ([[maybe_unused]] auto _ : std::views::iota(0uz, m_info.dry_run_size)) {
    m_engine.run(inference_info, cb);
    semaphore.acquire();
  }
}

auto BenchmarkState::task() -> void {
  try {
    latch.wait();

    while (true) {
      const auto run_idx = m_start_cnt.fetch_add(1, std::memory_order_relaxed);
      if (run_idx >= m_info.run_size) {
        break;
      }

      auto& run_info = m_run_infos[run_idx];
      auto& cb = *m_callback_pool.allocate();
      cb.data = &run_info;
      cb.func = &callback_func;
      run_info.callback = &cb;
      auto inference_info = get_inference_info(run_idx % m_info.batch_pool_size);
      run_info.t0 = std::chrono::steady_clock::now();
      m_engine.run(inference_info, cb);
    }
  } catch (...) {
    std::terminate();
  }
}

auto BenchmarkState::get_inference_info(std::size_t idx) -> InferenceInfo {
  return {
    .image_input = m_storage.image_input.batch(idx),
    .additional_input = m_storage.additional_input.batch(idx),
    .policy_output = m_storage.priors_output.batch(idx),
    .value_output = m_storage.value_output.batch(idx),
  };
}

auto benchmark_inference_engine(const InferenceEngineBenchmarkInfo& info)
  -> InferenceEngineBenchmarkResult {
  std::ifstream file{info.model_path, std::ios::binary};
  if (!file) {
    throw std::runtime_error("Failed to open model file: " + info.model_path);
  }
  auto model = create_model_from_stream(file, info.batch_size);

  BenchmarkState benchmark{info, std::move(model)};
  benchmark.run();
  return benchmark.results();
}

}  // namespace alpack
