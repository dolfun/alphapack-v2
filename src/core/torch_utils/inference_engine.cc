#include "inference_engine.h"

#include <atomic>
#include <exception>
#include <iostream>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>

namespace alpack {

InferenceEngine::InferenceEngine(InferenceModel model, size_t stream_pool_size)
    : m_model{std::move(model)}, m_curr_stream_idx{0} {
  if (stream_pool_size == 0) {
    throw std::invalid_argument("Pool size cannot be zero.");
  }

  m_streams.reserve(stream_pool_size);
  for (size_t i = 0; i < stream_pool_size; ++i) {
    m_streams.emplace_back(at::cuda::getStreamFromPool());
  }
}

auto InferenceEngine::get_next_stream() noexcept -> at::cuda::CUDAStream {
  size_t idx = m_curr_stream_idx.fetch_add(1, std::memory_order_relaxed);
  return m_streams[idx % m_streams.size()];
}

auto InferenceEngine::run(const InferenceInfo& input) -> InferenceResult {
  try {
    auto stream = get_next_stream();
    at::cuda::CUDAStreamGuard stream_guard{stream};

    m_model.infer(input);

    InferenceResult result{};
    result.m_event.record(stream);
    return result;

  } catch (const std::exception& e) {
    std::println(std::cerr, "Error: {}", e.what());
    std::terminate();
  }
}

}  // namespace alpack