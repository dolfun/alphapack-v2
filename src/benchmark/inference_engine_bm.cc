#include <core/inference/allocator.h>
#include <core/inference/inference_engine.h>
#include <core/mcts/model_adapter.h>
#include <core/memory/batched_array_pool.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <latch>
#include <mutex>
#include <print>
#include <random>
#include <semaphore>
#include <string>

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
  using HostPool = alpack::BatchedArrayPool<float, N, alpack::CudaHostAllocator>;

  HostPool<alpack::ModelAdapter::image_input_size> image_input;
  HostPool<alpack::ModelAdapter::additional_input_size> additional_input;
  HostPool<alpack::ModelAdapter::priors_output_size> priors_output;
  HostPool<alpack::ModelAdapter::value_output_size> value_output;
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
    auto mean = static_cast<double>(sum) / static_cast<double>(sample_count);
    return std::make_tuple(mean, max);
  }

private:
  std::size_t sample_count{0}, sum{0}, max{0};
};

struct BenchmarkInfo {
  std::size_t run_size;
  std::size_t dry_run_size;
  std::size_t batch_size;
  std::size_t batch_pool_size;
  std::size_t thread_pool_size;
  std::size_t stream_pool_size;
};

class BenchmarkState {
public:
  BenchmarkState(const BenchmarkInfo& info, alpack::InferenceModel model)
      : m_info{info},
        m_engine{std::move(model), m_info.stream_pool_size},
        m_storage{m_info.batch_size, m_info.batch_pool_size},
        m_callback_pool{m_info.batch_pool_size},
        m_run_infos(m_info.run_size),
        latch{1},
        m_start_cnt{0},
        m_finish_cnt{0} {
    std::ranges::for_each(m_run_infos, [this](auto& m_info) {
      m_info.callback_pool = &m_callback_pool;
      m_info.finish_cnt = &m_finish_cnt;
    });
  }

  auto run() -> void {
    dry_run();

    std::vector<std::jthread> threads;
    threads.reserve(m_info.thread_pool_size);
    std::generate_n(std::back_inserter(threads), m_info.thread_pool_size, [this] {
      return std::jthread{&BenchmarkState::task, this};
    });

    const auto benchmark_start = std::chrono::steady_clock::now();
    latch.count_down();
    while (true) {
      const auto progress = m_finish_cnt.load(std::memory_order_relaxed);
      auto percentage = (static_cast<double>(progress) * 100.0) / static_cast<double>(m_info.run_size);
      std::print("Progress: {:6.2f}%\r", percentage);
      if (progress >= m_info.run_size) break;

      m_in_flight_counter.update(m_callback_pool.in_flight_count());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::ranges::for_each(threads, &std::jthread::join);
    const auto benchmark_end = std::chrono::steady_clock::now();
    m_time_taken = benchmark_end - benchmark_start;
  }

  auto print_stats() const noexcept -> void;

private:
  auto dry_run() -> void;
  auto task() -> void;
  auto get_inference_info(std::size_t idx) -> alpack::InferenceInfo;

  static auto callback_func(void* data) -> void {
    auto& run_info = *static_cast<RunInfo*>(data);
    run_info.t1 = std::chrono::steady_clock::now();
    run_info.finish_cnt->fetch_add(1, std::memory_order_relaxed);
    run_info.callback_pool->free(run_info.callback);
  }

  BenchmarkInfo m_info;
  alpack::InferenceEngine m_engine;
  DataPool m_storage;
  ObjectPool<alpack::InferenceCallback> m_callback_pool;

  struct RunInfo {
    ObjectPool<alpack::InferenceCallback>* callback_pool{};
    alpack::InferenceCallback* callback{};
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

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::invalid_argument("Usage: {} <model_path> [batch_size] [thread_count] [stream_count]");
    }

    std::string model_path = argv[1];

    BenchmarkInfo benchmark_info{
      .run_size = 10'000,
      .dry_run_size = 64,
      .batch_size = argc >= 3 ? std::stoull(argv[2]) : 96,
      .batch_pool_size = 512,
      .thread_pool_size = argc >= 4 ? std::stoull(argv[3]) : 4,
      .stream_pool_size = argc >= 5 ? std::stoull(argv[4]) : 8
    };

    std::println("--- Inference Engine Benchmark ---");
    std::println("Model: {}", model_path);
    std::println("Batch Size: {}", benchmark_info.batch_size);
    std::println("Threads: {}", benchmark_info.thread_pool_size);
    std::println("Streams: {}", benchmark_info.stream_pool_size);
    std::println("Total Evaluations: {}", benchmark_info.run_size);

    std::ifstream file{model_path, std::ios::binary};
    if (!file) {
      throw std::runtime_error("Failed to open model file: " + model_path);
    }
    alpack::InferenceModel model{file};

    BenchmarkState benchmark{benchmark_info, std::move(model)};
    benchmark.run();
    benchmark.print_stats();

  } catch (const std::exception& e) {
    std::println(std::cerr, "Error: {}", e.what());
    return -1;
  }

  return 0;
}

auto BenchmarkState::print_stats() const noexcept -> void {
  double throughput = static_cast<double>(m_info.run_size) / m_time_taken.count();
  auto [avg_lat, std_lat, min_lat, max_lat, calculated_in_flight] = [&] {
    const auto latencies = m_run_infos | std::views::transform([](const RunInfo& cb) {
                             return std::chrono::duration<double, std::milli>(cb.t1 - cb.t0).count();
                           }) |
                           std::ranges::to<std::vector>();
    const auto size = static_cast<double>(latencies.size());
    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double mean = sum / size;
    const double sq_sum = std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0);
    const double std_dev = std::sqrt(std::max(0.0, sq_sum / size - mean * mean));
    const auto [min_it, max_it] = std::minmax_element(latencies.begin(), latencies.end());
    const double in_flight = throughput * (mean / 1000.0);
    return std::make_tuple(mean, std_dev, *min_it, *max_it, in_flight);
  }();

  auto [in_flight_mean, in_flight_max] = m_in_flight_counter.stats();

  std::println("--------------------------------");
  std::println("Results (Batch Size: {}):", m_info.batch_size);
  std::println("  Throughput:      {:.2f} batches/sec", throughput);
  std::println("  Time Taken:      {:.2f} sec", m_time_taken.count());
  std::println("  Batch Latency:");
  std::println("    Avg:           {:.2f} ms", avg_lat);
  std::println("    Std Dev:       {:.2f} ms", std_lat);
  std::println("    Min:           {:.2f} ms", min_lat);
  std::println("    Max:           {:.2f} ms", max_lat);
  std::println("  In-Flight Batches:");
  std::println("    Avg(Meas.):    {:.2f}", in_flight_mean);
  std::println("    Avg(Calc.):    {:.2f}", calculated_in_flight);
  std::println("    Max:           {}", in_flight_max);
  std::println("--------------------------------");
  std::println("Results (Single Evaluation):");
  std::println("  Throughput:      {:.2f} evals/sec", throughput * static_cast<double>(m_info.batch_size));
  std::println("  Avg Latency:     {:.4f} ms", avg_lat / static_cast<double>(m_info.batch_size));
  std::println("--------------------------------");
}

auto BenchmarkState::dry_run() -> void {
  std::binary_semaphore semaphore{0};
  alpack::InferenceCallback cb;
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
  } catch (const std::exception& e) {
    std::println(std::cerr, "Worker Error: {}", e.what());
    std::terminate();
  }
}

auto BenchmarkState::get_inference_info(std::size_t idx) -> alpack::InferenceInfo {
  return {
    .image_input = m_storage.image_input.batch(idx),
    .image_input_shape =
      {static_cast<std::int64_t>(m_info.batch_size),
       alpack::State::bin_length,
       alpack::State::bin_length,
       alpack::ModelAdapter::input_feature_count},
    .additional_input = m_storage.additional_input.batch(idx),
    .additional_input_shape =
      {static_cast<std::int64_t>(m_info.batch_size), alpack::ModelAdapter::additional_input_count},
    .policy_output = m_storage.priors_output.batch(idx),
    .policy_output_shape = {static_cast<std::int64_t>(m_info.batch_size), alpack::State::bin_base_size},
    .value_output = m_storage.value_output.batch(idx),
    .value_output_shape = {static_cast<std::int64_t>(m_info.batch_size), alpack::ModelAdapter::value_support_count},
  };
}