#include "inference_model.h"

// Do something about this?
#pragma warning(push)
#pragma warning(disable : 4267 4702)
#include <torch/script.h>
#pragma warning(pop)

#include <cassert>
#include <sstream>

namespace alpack {

struct InferenceModel::Impl {
  torch::jit::Module m_model;

  static auto load_model(std::istream& in) -> torch::jit::Module {
    return torch::jit::load(in, torch::kCUDA, false);
  }

  explicit Impl(std::istream& in) : m_model{load_model(in)} {}

  explicit Impl(const std::string& data) {
    std::istringstream iss{data};
    m_model = load_model(iss);
  }
};

InferenceModel::InferenceModel(std::istream& in) : m_pimpl{std::make_unique<Impl>(in)} {}
InferenceModel::InferenceModel(const std::string& data) : m_pimpl{std::make_unique<Impl>(data)} {}

InferenceModel::~InferenceModel() = default;

InferenceModel::InferenceModel(InferenceModel&&) noexcept = default;
InferenceModel& InferenceModel::operator=(InferenceModel&&) noexcept = default;

auto InferenceModel::infer(const InferenceInfo& info) const -> void {
  c10::InferenceMode inference_mode_guard;

  const auto options = torch::TensorOptions{}.dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);

  // Copy input to CPU
  const auto image_input_cpu = torch::from_blob(info.image_input_ptr, info.image_input_shape, options);
  assert(image_input_cpu.is_pinned());
  auto image_input_gpu = image_input_cpu.to(torch::kCUDA, true);

  const auto additional_input_cpu = torch::from_blob(info.additional_input_ptr, info.additional_input_shape, options);
  assert(additional_input_cpu.is_pinned());
  auto additional_input_gpu = additional_input_cpu.to(torch::kCUDA, true);

  // Inference
  assert(!m_pimpl->m_model.is_training());
  const auto output = m_pimpl->m_model.forward({image_input_gpu, additional_input_gpu});

  // Extract output
  const auto output_tuple = output.toTupleRef();
  const auto& elements = output_tuple.elements();
  const auto policy_output_gpu = elements[0].toTensor();
  const auto value_output_gpu = elements[1].toTensor();

  // Copy output to CPU
  const auto policy_output_cpu = torch::from_blob(info.policy_output_ptr, info.policy_output_shape, options);
  assert(policy_output_cpu.is_pinned());
  (void)policy_output_cpu.copy_(policy_output_gpu, true);

  const auto value_output_cpu = torch::from_blob(info.value_output_ptr, info.value_output_shape, options);
  assert(value_output_cpu.is_pinned());
  (void)value_output_cpu.copy_(value_output_gpu, true);
}

}  // namespace alpack