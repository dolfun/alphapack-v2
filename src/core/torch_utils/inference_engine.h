#pragma once
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <core/torch_utils/inference_model.h>

namespace alpack {

class InferenceCallback {
public:
  void (*func)(void*){};
  void* data{};

private:
  friend class InferenceEngine;

  at::cuda::CUDAEvent m_event{cudaEventDisableTiming};
};

class InferenceEngine {
public:
  InferenceEngine(InferenceModel model, std::size_t stream_pool_size);

  auto run(const InferenceInfo&, InferenceCallback&) -> void;

private:
  InferenceModel m_model;
  std::atomic<std::size_t> m_curr_stream_idx;
  at::cuda::CUDAStream m_notify_stream;
  std::vector<at::cuda::CUDAStream> m_worker_streams;
};

}  // namespace alpack