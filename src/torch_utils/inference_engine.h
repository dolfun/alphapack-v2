#pragma once
#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <vector>

#include "inference_model.h"

namespace torch_utils {

class InferenceEngine {
public:
  InferenceEngine(InferenceModel model, size_t pool_size);

  ~InferenceEngine() = default;

  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;
  InferenceEngine(InferenceEngine&&) = delete;
  InferenceEngine& operator=(InferenceEngine&&) = delete;

  [[nodiscard]] auto run(const InferenceInfo&) -> InferenceResult;

private:
  [[nodiscard]] auto get_next_stream() noexcept -> at::cuda::CUDAStream;

  InferenceModel m_model;
  std::atomic<size_t> m_curr_stream_idx;
  std::vector<at::cuda::CUDAStream> m_streams;
};

}  // namespace torch_utils