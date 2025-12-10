#include "inference_model.h"

#include <c10/cuda/CUDAGuard.h>

#include <cassert>
#include <sstream>

namespace torch_utils {

InferenceModel::InferenceModel(torch::jit::script::Module model) : m_model{ std::move(model) } {
  m_model.eval();
  m_model = torch::jit::freeze(m_model);
  m_model = torch::jit::optimize_for_inference(m_model);
}

InferenceModel::InferenceModel(std::istream& in)
    : InferenceModel{ torch::jit::load(in, torch::kCUDA, false) } {}

auto InferenceModel::make_from_bytes(const std::string& buffer) -> InferenceModel {
  std::istringstream stream(buffer);
  return InferenceModel(stream);
}

auto InferenceModel::run(const InferenceInfo& info, const at::cuda::CUDAStream& stream)
  -> InferenceResult {
  at::cuda::CUDAStreamGuard stream_guard{ stream };
  at::InferenceMode inference_mode_guard;

  auto options =
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);

  // Copy input to CPU
  auto input_cpu = torch::from_blob(info.input_ptr, info.input_shape, options);
  assert(input_cpu.is_pinned());
  auto input_gpu = input_cpu.to(torch::kCUDA, true);

  // Inference
  auto output = m_model.forward({ input_gpu });

  // Extract output
  auto output_tuple = output.toTupleRef();
  const auto& elements = output_tuple.elements();
  auto policy_output_gpu = elements[0].toTensor();
  auto value_output_gpu = elements[1].toTensor();

  // Copy output to CPU
  auto policy_output_cpu =
    torch::from_blob(info.policy_output_ptr, info.policy_output_shape, options);
  assert(policy_output_cpu.is_pinned());
  policy_output_cpu.copy_(policy_output_gpu, true);

  auto value_output_cpu = torch::from_blob(info.value_output_ptr, info.value_output_shape, options);
  assert(value_output_cpu.is_pinned());
  value_output_cpu.copy_(value_output_gpu, true);

  // Event
  at::cuda::CUDAEvent event{ cudaEventDisableTiming };
  event.record(stream);

  // Result
  return InferenceResult{ std::move(event) };
}

}  // namespace torch_utils