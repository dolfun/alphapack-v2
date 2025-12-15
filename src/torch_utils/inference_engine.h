#pragma once
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "inference_model.h"

namespace torch_utils {

class InferenceResult {
public:
  [[nodiscard]] auto is_done() const noexcept -> bool {
    return m_event.query();
  }

private:
  friend class InferenceEngine;

  at::cuda::CUDAEvent m_event{cudaEventDisableTiming};
};

class InferenceEngine {
public:
  InferenceEngine(InferenceModel model, size_t stream_pool_size);

  [[nodiscard]] auto run(const InferenceInfo&) -> InferenceResult;

private:
  [[nodiscard]] auto get_next_stream() noexcept -> at::cuda::CUDAStream;

  InferenceModel m_model;
  std::atomic<size_t> m_curr_stream_idx;
  std::vector<at::cuda::CUDAStream> m_streams;
};

}  // namespace torch_utils