#pragma once
#include <core/inference/inference_model.h>
#include <core/state/state.h>

#include <cstdint>
#include <span>

namespace alpack {

struct ModelAdapter {
  static constexpr std::size_t input_feature_count = 2;
  static constexpr std::size_t additional_input_count = 4 * State::max_item_count;
  static constexpr std::size_t value_support_count = 101;

  static constexpr std::size_t image_input_size = input_feature_count * State::bin_base_size;
  static constexpr std::size_t additional_input_size = additional_input_count;
  static constexpr std::size_t priors_output_size = State::action_count;
  static constexpr std::size_t value_output_size = value_support_count;

  static constexpr std::size_t transform_count = 8;

  static constexpr auto scalar_type = ScalarType::bfloat16;

  struct DecodeInfo {
    std::uint8_t K;
    Item::dim_type item_length, item_width;
    bool final_state;
  };

  static auto encode(
    const State& state,
    std::uint8_t K,
    std::span<float, image_input_size> image_data_out,
    std::span<float, additional_input_size> additional_data_out
  ) noexcept -> DecodeInfo;

  static auto decode(
    DecodeInfo decode_info,
    std::span<const float, priors_output_size> priors_data_in,
    std::span<const float, value_output_size> value_data_in,
    std::span<float, State::action_count> priors_out,
    float& value_out
  ) noexcept -> void;
};

auto make_inference_model(std::istream&, std::size_t batch_size) -> InferenceModel;

}  // namespace alpack