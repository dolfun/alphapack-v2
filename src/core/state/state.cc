#include "state.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <ranges>

namespace alpack {

template <typename T, std::size_t box_n, std::size_t box_m>
constexpr auto
get_2d_window_max_in_box(const NdArray<T, box_n, box_m>& box, std::size_t window_n, std::size_t window_m) {
  NdArray<T, box_n, box_m> result{};

  for (std::size_t x = 0; x < box_n; ++x) {
    for (std::size_t y = 0; y <= box_m - window_m; ++y) {
      T current_max = box[x, y];
      for (std::size_t k = 1; k < window_m; ++k) {
        current_max = std::max(current_max, box[x, y + k]);
      }
      result[x, y] = current_max;
    }
  }

  for (std::size_t y = 0; y <= box_m - window_m; ++y) {
    for (std::size_t x = 0; x <= box_n - window_n; ++x) {
      T current_max = result[x, y];
      for (std::size_t k = 1; k < window_n; ++k) {
        current_max = std::max(current_max, result[x + k, y]);
      }
      result[x, y] = current_max;
    }
  }

  return result;
}

auto State::feasibility_mask() const noexcept -> FeasibilityMask {
  FeasibilityMask mask{};
  std::ranges::transform(m_feasibility_info, mask.begin(), [](auto height) {
    return height != invalid_feasible_height;
  });
  return mask;
}

auto State::packing_efficiency() const noexcept -> float {
  auto get_packed_volume = [](const Item item) { return item.placed ? item.volume() : 0; };
  const auto packed_volume =
    std::ranges::fold_left(m_items | std::views::transform(get_packed_volume), 0, std::plus<>{});
  constexpr auto bin_volume = bin_base_size * bin_height;
  return static_cast<float>(packed_volume) / bin_volume;
}

auto State::feasible_actions() const noexcept -> std::vector<Action> {
  if (m_items.front().placed) return {};

  std::vector<Action> actions;
  for (const auto [idx, height] : std::views::enumerate(m_feasibility_info)) {
    if (height != invalid_feasible_height) {
      actions.push_back(static_cast<Action>(idx));
    }
  }

  return actions;
}

auto State::transition(Action action_idx) -> float {
  std::ranges::rotate(m_items, m_items.begin() + 1);
  Item& selected_item = m_items.back();
  selected_item.placed = true;

  const auto x0 = action_idx / bin_length, y0 = action_idx % bin_length;
  const auto base_height = m_feasibility_info[x0, y0];
  if (action_idx >= action_count || base_height == invalid_feasible_height) {
    throw std::runtime_error("Invalid Action");
  }

  for (auto x = x0; x < x0 + selected_item.shape.x; ++x) {
    for (auto y = y0; y < y0 + selected_item.shape.y; ++y) {
      m_height_map[x, y] = base_height + selected_item.shape.z;
    }
  }

  update_feasibility_info_with_front_item();

  constexpr auto reward_scaling = max_item_count * (max_item_count + 1) / 2;
  const auto used_items_count =
    std::ranges::count(m_items, true, [](const Item item) { return item.volume() > 0 && item.placed; });
  const auto reward = static_cast<float>(used_items_count) / reward_scaling;
  return reward;
}

auto State::update_feasibility_info_with_front_item() noexcept -> void {
  m_feasibility_info.fill(invalid_feasible_height);

  const auto front_item = m_items.front();
  if (front_item.placed) return;

  auto max_height_arr = get_2d_window_max_in_box(m_height_map, front_item.shape.x, front_item.shape.y);
  for (std::size_t x = 0; x <= m_feasibility_info.shape[0] - front_item.shape.x; ++x) {
    for (std::size_t y = 0; y <= m_feasibility_info.shape[1] - front_item.shape.y; ++y) {
      const auto max_height = max_height_arr[x, y];
      if (max_height + front_item.shape.z <= bin_height) {
        m_feasibility_info[x, y] = max_height;
      }
    }
  }
}

}  // namespace alpack
