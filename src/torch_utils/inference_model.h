#pragma once
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>

#include <array>
#include <istream>
#include <string>

namespace torch_utils {

struct InferenceInfo {
  std::array<int64_t, 4> input_shape;
  void* input_ptr;

  std::array<int64_t, 3> policy_output_shape;
  void* policy_output_ptr;

  std::array<int64_t, 2> value_output_shape;
  void* value_output_ptr;
};

class InferenceResult {
public:
  InferenceResult() = default;

  [[nodiscard]] auto is_done() const noexcept -> bool {
    return m_event.query();
  }

private:
  friend class InferenceModel;

  InferenceResult(at::cuda::CUDAEvent event) : m_event{ std::move(event) } {}

  at::cuda::CUDAEvent m_event{};
};

class InferenceModel {
public:
  InferenceModel(torch::jit::script::Module);
  ~InferenceModel() = default;

  InferenceModel(const InferenceModel&) = delete;
  InferenceModel& operator=(const InferenceModel&) = delete;
  InferenceModel(InferenceModel&&) = default;
  InferenceModel& operator=(InferenceModel&&) = default;

  static auto make_from_bytes(const std::string&) -> InferenceModel;

  [[nodiscard]] auto run(const InferenceInfo&, const at::cuda::CUDAStream&) -> InferenceResult;

private:
  InferenceModel(std::istream&);

  torch::jit::script::Module m_model;
};

};  // namespace torch_utils