#pragma once
#include <array>
#include <memory>
#include <span>
#include <string>

namespace alpack {

struct InferenceInfo {
  std::span<float> image_input;
  std::array<int64_t, 4> image_input_shape;

  std::span<float> additional_input;
  std::array<int64_t, 2> additional_input_shape;

  std::span<float> policy_output;
  std::array<int64_t, 2> policy_output_shape;

  std::span<float> value_output;
  std::array<int64_t, 2> value_output_shape;
};

class InferenceModel {
public:
  explicit InferenceModel(std::istream&);
  explicit InferenceModel(const std::string&);

  ~InferenceModel();

  InferenceModel(const InferenceModel&) = delete;
  InferenceModel& operator=(const InferenceModel&) = delete;
  InferenceModel(InferenceModel&&) noexcept;
  InferenceModel& operator=(InferenceModel&&) noexcept;

  auto infer(const InferenceInfo&) const -> void;

private:
  struct Impl;
  std::unique_ptr<Impl> m_pimpl;
};

}  // namespace alpack