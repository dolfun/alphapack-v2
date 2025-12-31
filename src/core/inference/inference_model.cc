#include "inference_model.h"

// Do something about this?
#pragma warning(push)
#pragma warning(disable : 4267 4702)
#include <torch/script.h>
#pragma warning(pop)

namespace alpack {

namespace {

template <typename From, std::size_t N>
constexpr auto to_ssize_array(const std::array<From, N>& src) {
  std::array<std::int64_t, N> dest{};
  std::ranges::copy(src, dest.begin());
  return dest;
}

auto to_torch_scalar_type(ScalarType type) -> c10::ScalarType {
  switch (type) {
    case ScalarType::float32:
      return torch::kFloat32;
    case ScalarType::float16:
      return torch::kFloat16;
    case ScalarType::bfloat16:
      return torch::kBFloat16;
    default:
      std::unreachable();
  }
}

}  // namespace

struct InferenceModel::Impl {
  explicit Impl(std::istream& in) : m_model{torch::jit::load(in, torch::kCUDA, false)} {}

  torch::jit::Module m_model;
};

InferenceModel::InferenceModel(std::istream& in, const ModelCreateInfo& info)
    : m_pimpl{std::make_unique<Impl>(in)},
      m_scalar_type{info.scalar_type},
      m_image_input_shape{to_ssize_array(info.image_input_shape)},
      m_additional_input_shape{to_ssize_array(info.additional_input_shape)},
      m_policy_output_shape{to_ssize_array(info.policy_output_shape)},
      m_value_output_shape{to_ssize_array(info.value_output_shape)} {}

InferenceModel::~InferenceModel() = default;
InferenceModel::InferenceModel(InferenceModel&&) noexcept = default;
InferenceModel& InferenceModel::operator=(InferenceModel&&) noexcept = default;

auto InferenceModel::infer(const InferenceInfo& info) const -> void {
  c10::InferenceMode inference_mode_guard;

  const auto cpu_options = torch::TensorOptions{}.dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
  const auto gpu_options = torch::TensorOptions{}.dtype(to_torch_scalar_type(m_scalar_type)).device(torch::kCUDA);

  // Copy input to CPU
  const auto image_input_cpu =
    torch::from_blob(info.image_input.data(), m_image_input_shape, cpu_options).permute({0, 3, 1, 2});
  auto image_input_gpu = image_input_cpu.to(gpu_options, true, false, torch::MemoryFormat::ChannelsLast);

  const auto additional_input_cpu =
    torch::from_blob(info.additional_input.data(), m_additional_input_shape, cpu_options);
  auto additional_input_gpu = additional_input_cpu.to(gpu_options, true);

  // Inference
  const auto output = m_pimpl->m_model.forward({image_input_gpu, additional_input_gpu});

  // Extract output
  const auto& output_tuple = output.toTupleRef();
  const auto& elements = output_tuple.elements();
  const auto policy_output_gpu = elements[0].toTensor();
  const auto value_output_gpu = elements[1].toTensor();

  // Copy output to CPU
  const auto policy_output_cpu = torch::from_blob(info.policy_output.data(), m_policy_output_shape, cpu_options);
  (void)policy_output_cpu.copy_(policy_output_gpu, true);

  const auto value_output_cpu = torch::from_blob(info.value_output.data(), m_value_output_shape, cpu_options);
  (void)value_output_cpu.copy_(value_output_gpu, true);
}

}  // namespace alpack