#pragma once
#include <array>
#include <memory>
#include <span>

namespace alpack {

enum class ScalarType : std::uint8_t { float32, float16, bfloat16 };

struct ModelCreateInfo {
  ScalarType scalar_type{};
  std::array<std::size_t, 4> image_input_shape{};
  std::array<std::size_t, 2> additional_input_shape{};
  std::array<std::size_t, 2> policy_output_shape{};
  std::array<std::size_t, 2> value_output_shape{};
};

struct InferenceInfo {
  std::span<float> image_input;
  std::span<float> additional_input;
  std::span<float> policy_output;
  std::span<float> value_output;
};

class InferenceModel {
public:
  InferenceModel(std::istream& in, const ModelCreateInfo& info);

  ~InferenceModel();

  InferenceModel(const InferenceModel&) = delete;
  InferenceModel& operator=(const InferenceModel&) = delete;
  InferenceModel(InferenceModel&&) noexcept;
  InferenceModel& operator=(InferenceModel&&) noexcept;

  auto infer(const InferenceInfo& info) const -> void;

private:
  struct Impl;
  std::unique_ptr<Impl> m_pimpl;

  ScalarType m_scalar_type;
  std::array<std::int64_t, 4> m_image_input_shape;
  std::array<std::int64_t, 2> m_additional_input_shape{};
  std::array<std::int64_t, 2> m_policy_output_shape{};
  std::array<std::int64_t, 2> m_value_output_shape{};
};

}  // namespace alpack