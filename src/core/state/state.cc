#include "state.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <ranges>

namespace alpack {

template <typename T, size_t box_n, size_t box_m>
constexpr auto get_2d_window_max_in_box(const NdArray<T, box_n, box_m>& box, size_t window_n, size_t window_m) {
  NdArray<T, box_n, box_m> result{};

  for (size_t x = 0; x < box_n; ++x) {
    for (size_t y = 0; y <= box_m - window_m; ++y) {
      T current_max = box[x, y];
      for (size_t k = 1; k < window_m; ++k) {
        current_max = std::max(current_max, box[x, y + k]);
      }
      result[x, y] = current_max;
    }
  }

  for (size_t y = 0; y <= box_m - window_m; ++y) {
    for (size_t x = 0; x <= box_n - window_n; ++x) {
      T current_max = result[x, y];
      for (size_t k = 1; k < window_n; ++k) {
        current_max = std::max(current_max, result[x + k, y]);
      }
      result[x, y] = current_max;
    }
  }

  return result;
}

auto State::feasibility_mask() const noexcept -> FeasibilityMask {
  FeasibilityMask mask{};
  std::ranges::transform(m_feasibility_info, mask.begin(), [](auto val) { return val >= 0; });
  return mask;
}

auto State::packing_efficiency() const noexcept -> float {
  auto get_packed_volume = [](const Item& item) { return item.placed ? item.volume() : 0u; };
  const auto packed_volume =
    std::ranges::fold_left(m_items | std::views::transform(get_packed_volume), 0u, std::plus<>{});
  constexpr auto bin_volume = bin_base_size * bin_height;
  return static_cast<float>(packed_volume) / bin_volume;
}

auto State::feasible_actions() const noexcept -> std::vector<Action> {
  if (m_items.front().placed) return {};

  Action action_idx = 0;
  std::vector<Action> actions;
  for (const auto feasible : m_feasibility_info) {
    if (feasible >= 0) {
      actions.push_back(action_idx);
    }
    ++action_idx;
  }

  return actions;
}

auto State::transition(Action action_idx) noexcept -> float {
  assert(action_idx < action_count);
  assert(!m_items.front().placed);

  std::ranges::rotate(m_items, m_items.begin() + 1);
  Item& selected_item = m_items.back();
  selected_item.placed = true;

  const auto x0 = action_idx / bin_length, y0 = action_idx % bin_length;
  const auto base_height = m_feasibility_info[x0, y0];
  assert(base_height >= 0);

  for (auto x = x0; x < x0 + selected_item.shape.x; ++x) {
    for (auto y = y0; y < y0 + selected_item.shape.y; ++y) {
      m_height_map[x, y] = base_height + selected_item.shape.z;
    }
  }

  update_feasibility_info(m_items.front());

  constexpr auto reward_scaling = max_item_count * (max_item_count + 1) / 2;
  const auto used_items_count =
    std::ranges::count(m_items, true, [](const Item& item) { return item.volume() > 0 && item.placed; });
  const auto reward = static_cast<float>(used_items_count) / reward_scaling;
  return reward;
}

auto State::update_feasibility_info(const Item& item) noexcept -> void {
  m_feasibility_info.fill(-1);
  if (item.placed) return;

  auto max_height_arr = get_2d_window_max_in_box(m_height_map, item.shape.x, item.shape.y);
  for (size_t x = 0; x <= m_feasibility_info.shape[0] - item.shape.x; ++x) {
    for (size_t y = 0; y <= m_feasibility_info.shape[1] - item.shape.y; ++y) {
      if (const auto max_height = max_height_arr[x, y]; max_height + item.shape.z <= static_cast<int8_t>(bin_height)) {
        m_feasibility_info[x, y] = static_cast<int8_t>(max_height);
      }
    }
  }
}

}  // namespace alpack
