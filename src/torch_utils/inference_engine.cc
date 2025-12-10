#include "inference_engine.h"

#include <stdexcept>
#include <utility>

namespace torch_utils {

InferenceEngine::InferenceEngine(InferenceModel model, size_t pool_size)
    : m_model{ std::move(model) }, m_curr_stream_idx{ 0 } {
  if (pool_size == 0) {
    throw std::invalid_argument("Pool size cannot be zero.");
  }

  m_streams.reserve(pool_size);
  for (size_t i = 0; i < pool_size; ++i) {
    m_streams.emplace_back(at::cuda::getStreamFromPool());
  }
};

auto InferenceEngine::run(const InferenceInfo& input) -> InferenceResult {
  auto stream = get_next_stream();
  return m_model.run(input, stream);
}

auto InferenceEngine::get_next_stream() noexcept -> at::cuda::CUDAStream {
  size_t idx = m_curr_stream_idx.fetch_add(1, std::memory_order_relaxed);
  return m_streams[idx % m_streams.size()];
}

}  // namespace torch_utils