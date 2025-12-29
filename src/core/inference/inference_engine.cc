#include "inference_engine.h"

#include <c10/cuda/CUDAGuard.h>

#include <atomic>
#include <exception>
#include <iostream>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>

namespace alpack {

InferenceEngine::InferenceEngine(InferenceModel model, std::size_t stream_pool_size)
    : m_model{std::move(model)}, m_curr_stream_idx{0}, m_notify_stream{at::cuda::getStreamFromPool()} {
  if (stream_pool_size == 0) {
    throw std::invalid_argument("Pool size cannot be zero.");
  }

  m_worker_streams.reserve(stream_pool_size);
  for (std::size_t i = 0; i < stream_pool_size; ++i) {
    m_worker_streams.emplace_back(at::cuda::getStreamFromPool());
  }
}

auto InferenceEngine::run(const InferenceInfo& input, InferenceCallback& callback) -> void {
  try {
    const auto idx = m_curr_stream_idx.fetch_add(1, std::memory_order_relaxed);
    const auto stream = m_worker_streams[idx % m_worker_streams.size()];
    at::cuda::CUDAStreamGuard stream_guard{stream};

    m_model.infer(input);

    callback.m_event.record(stream);
    cudaStreamWaitEvent(m_notify_stream, callback.m_event);
    cudaLaunchHostFunc(m_notify_stream, callback.func, callback.data);

  } catch (const std::exception& e) {
    std::println(std::cerr, "Error: {}", e.what());
    std::terminate();
  }
}

}  // namespace alpack