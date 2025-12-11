#pragma once

#include <array>
#include <istream>
#include <memory>
#include <string>

namespace torch_utils {

struct InferenceInfo {
  std::array<int64_t, 4> image_input_shape;
  void* image_input_ptr;

  std::array<int64_t, 2> additional_input_shape;
  void* additional_input_ptr;

  std::array<int64_t, 2> policy_output_shape;
  void* policy_output_ptr;

  std::array<int64_t, 2> value_output_shape;
  void* value_output_ptr;
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

  auto infer(const InferenceInfo&) -> void;

private:
  struct Impl;
  std::unique_ptr<Impl> m_pimpl;
};

}  // namespace torch_utils