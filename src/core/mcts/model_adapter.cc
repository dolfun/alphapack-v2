#include "model_adapter.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace alpack {

template <std::size_t L>
static auto transform(
  ModelAdapter::Transform K,
  std::size_t x,
  std::size_t y,
  std::size_t l,
  std::size_t w
) noexcept -> std::pair<std::size_t, std::size_t> {
  using enum ModelAdapter::Transform;
  switch (K) {
    case Identity:
      return {x, y};
    case Rotate90:
      return {L - y - w, x};
    case Rotate180:
      return {L - x - l, L - y - w};
    case Rotate270:
      return {y, L - x - l};
    case FlipHorizontal:
      return {L - x - l, y};
    case FlipAntiDiagonal:
      return {L - y - w, L - x - l};
    case FlipVertical:
      return {x, L - y - w};
    case FlipMainDiagonal:
      return {y, x};
    default:
      std::unreachable();
  }
}

template <std::size_t N>
static auto apply_softmax(std::span<float, N> arr) noexcept -> void {
  const float max_val = std::ranges::max(arr);
  float sum{};
  std::ranges::for_each(arr, [max_val, &sum](float& x) {
    x = std::exp(x - max_val);
    sum += x;
  });
  constexpr float eps = 1e-7f;
  if (sum > eps) [[likely]] {
    std::ranges::for_each(arr, [sum](float& x) { x /= sum; });
  }
}

auto ModelAdapter::encode(
  const State& state,
  Transform K,
  std::span<float, image_input_size> image_data_out,
  std::span<float, additional_input_size> additional_data_out
) noexcept -> DecodeInfo {
  constexpr auto L = State::bin_length;
  const auto curr_item = state.items().front();
  const std::size_t l = curr_item.shape.x;
  const std::size_t w = curr_item.shape.y;

  const auto& height_map_in = state.height_map();
  for (std::size_t x = 0; x < L; ++x) {
    for (std::size_t y = 0; y < L; ++y) {
      const auto [x_t, y_t] = transform<L>(K, x, y, 1, 1);
      const auto idx_t = input_feature_count * (x_t * L + y_t);
      image_data_out[idx_t] = static_cast<float>(height_map_in[x, y]) / State::bin_height;
      image_data_out[idx_t + 1] = 0.0f;
    }
  }

  const auto& feasibility_info_in = state.feasibility_info();
  bool has_valid_placement = false;
  if (!curr_item.placed) [[likely]] {
    for (std::size_t x = 0; x <= L - l; ++x) {
      for (std::size_t y = 0; y <= L - w; ++y) {
        const auto [x_t, y_t] = transform<L>(K, x, y, l, w);
        const auto idx_t = (input_feature_count * (x_t * L + y_t)) + 1;
        const bool is_valid_placement = feasibility_info_in[x, y] != State::invalid_feasible_height;
        has_valid_placement |= is_valid_placement;
        image_data_out[idx_t] = static_cast<float>(is_valid_placement);
      }
    }
  }

  const bool to_flip = std::to_underlying(K) % 2 == 1;
  auto it = additional_data_out.begin();
  for (auto [shape, is_placed] : state.items()) {
    it[0] = static_cast<float>(to_flip ? shape.y : shape.x) / State::bin_length;
    it[1] = static_cast<float>(to_flip ? shape.x : shape.y) / State::bin_length;
    it[2] = static_cast<float>(shape.z) / State::bin_height;
    it[3] = (is_placed ? 1.0f : 0.0f);
    it += 4;
  }

  return DecodeInfo{
    .K = K,
    .item_length = static_cast<Item::dim_type>(l),
    .item_width = static_cast<Item::dim_type>(w),
    .final_state = curr_item.placed || !has_valid_placement
  };
}

auto ModelAdapter::decode(
  DecodeInfo decode_info,
  std::span<const float, priors_output_size> priors_data_in,
  std::span<const float, value_output_size> value_data_in,
  std::span<float, State::action_count> priors_out,
  float& value_out
) noexcept -> void {
  constexpr auto L = State::bin_length;
  const auto [K, l, w, final_state] = decode_info;

  if (final_state) [[unlikely]] {
    std::ranges::fill(priors_out, 0.0f);
    value_out = 0.0f;
    return;
  }

  constexpr float neg_infinity = -1e9f;
  std::ranges::fill(priors_out, neg_infinity);
  for (std::size_t x = 0; x <= L - l; ++x) {
    for (std::size_t y = 0; y <= L - w; ++y) {
      const auto [x_t, y_t] = transform<L>(K, x, y, l, w);
      const auto idx = (x * L) + y;
      const auto idx_t = (x_t * L) + y_t;
      priors_out[idx] = priors_data_in[idx_t];
    }
  }
  apply_softmax(priors_out);

  std::array<float, value_support_count> value_data{};
  std::ranges::copy(value_data_in, value_data.begin());
  apply_softmax(std::span(value_data));

  value_out = 0.0f;
  for (const auto [idx, val] : std::views::enumerate(value_data)) {
    value_out += val * static_cast<float>(idx) / (value_data.size() - 1);
  }
}

auto create_model_from_stream(std::istream& in, std::size_t batch_size) -> InferenceModel {
  const ModelCreateInfo info{
    .scalar_type = ModelAdapter::scalar_type,
    .image_input_shape =
      {batch_size, State::bin_length, State::bin_length, ModelAdapter::input_feature_count},
    .additional_input_shape = {batch_size, ModelAdapter::additional_input_count},
    .policy_output_shape = {batch_size, State::action_count},
    .value_output_shape = {batch_size, ModelAdapter::value_support_count}
  };
  return InferenceModel{in, info};
}

}  // namespace alpack