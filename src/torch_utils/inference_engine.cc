#include "inference_engine.h"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <exception>
#include <iostream>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>


namespace torch_utils {

struct InferenceResult::Impl {
  at::cuda::CUDAEvent m_event{cudaEventDisableTiming};
};

InferenceResult::InferenceResult() : m_pimpl{std::make_unique<Impl>()} {}
InferenceResult::~InferenceResult() = default;

InferenceResult::InferenceResult(InferenceResult&&) noexcept = default;
InferenceResult& InferenceResult::operator=(InferenceResult&&) noexcept = default;

auto InferenceResult::is_done() const noexcept -> bool {
  if (!m_pimpl) return false;
  return m_pimpl->m_event.query();
}

struct InferenceEngine::Impl {
  Impl(InferenceModel model, size_t pool_size) : m_model{std::move(model)}, m_curr_stream_idx{0} {
    if (pool_size == 0) {
      throw std::invalid_argument("Pool size cannot be zero.");
    }

    m_streams.reserve(pool_size);
    for (size_t i = 0; i < pool_size; ++i) {
      m_streams.emplace_back(at::cuda::getStreamFromPool());
    }
  }

  [[nodiscard]] auto get_next_stream() noexcept -> at::cuda::CUDAStream {
    size_t idx = m_curr_stream_idx.fetch_add(1, std::memory_order_relaxed);
    return m_streams[idx % m_streams.size()];
  }

  InferenceModel m_model;
  std::atomic<size_t> m_curr_stream_idx;
  std::vector<at::cuda::CUDAStream> m_streams;
};

InferenceEngine::InferenceEngine(InferenceModel model, size_t pool_size)
    : m_pimpl(std::make_unique<Impl>(std::move(model), pool_size)) {}

InferenceEngine::~InferenceEngine() = default;

auto InferenceEngine::run(const InferenceInfo& input) -> InferenceResult {
  try {
    auto stream = m_pimpl->get_next_stream();
    at::cuda::CUDAStreamGuard stream_guard{stream};

    m_pimpl->m_model.infer(input);

    InferenceResult result{};
    result.m_pimpl->m_event.record(stream);

    return result;

  } catch (const std::exception& e) {
    std::println(std::cerr, "Error: {}", e.what());
    std::terminate();
  }
}

}  // namespace torch_utils