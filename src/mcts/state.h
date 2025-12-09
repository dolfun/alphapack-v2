#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "item.h"
#include "ndarray.h"

namespace mcts {

template <typename T>
class Serializer;

template <typename R, typename T>
concept RangeOf =
  std::ranges::input_range<R> && std::convertible_to<std::ranges::range_value_t<R>, T>;

class State {
public:
  static constexpr size_t max_item_count = 64;
  static constexpr size_t bin_length = 10;
  static constexpr size_t bin_height = 10;
  static constexpr uint8_t action_count = bin_length * bin_length;

  template <typename T>
  using Array2D = NdArray<T, bin_length, bin_length>;

  using Items = std::array<Item, max_item_count>;
  using HeightMap = Array2D<uint8_t>;
  using FeasibilityInfo = Array2D<int8_t>;
  using FeasibilityMask = Array2D<bool>;

  using Action = uint8_t;
  static_assert(std::numeric_limits<Action>::max() >= action_count - 1);

  template <RangeOf<Item> ItemRange>
  explicit State(const ItemRange&);

  [[nodiscard]] auto items() const noexcept -> const Items&;
  [[nodiscard]] auto height_map() const noexcept -> const HeightMap&;
  [[nodiscard]] auto feasibility_info() const noexcept -> const FeasibilityInfo&;

  [[nodiscard]] auto feasibility_mask() const noexcept -> FeasibilityMask;
  [[nodiscard]] auto packing_efficiency() const noexcept -> float;
  [[nodiscard]] auto feasible_actions() const noexcept -> std::vector<Action>;

  [[nodiscard]] auto transition(Action) noexcept -> float;

private:
  State() = default;
  friend class Serializer<State>;

  auto update_feasibility_info(const Item&) noexcept -> void;

  Items m_items{};
  HeightMap m_height_map{};
  FeasibilityInfo m_feasibility_info{};
};

template <RangeOf<Item> ItemRange>
State::State(const ItemRange& items) {
  if (std::ranges::empty(items) || std::ranges::size(items) > max_item_count) {
    throw std::runtime_error("Invalid number of items in constructor");
  }

  std::ranges::for_each(items, [](const Item& item) {
    if (item.volume() == 0 || item.shape.x > bin_length || item.shape.y > bin_length ||
        item.shape.z > bin_height || item.placed) {
      throw std::runtime_error(
        std::format(
          "Invalid item in constructor: shape({}, {}, {}), placed({})",
          item.shape.x,
          item.shape.y,
          item.shape.z,
          item.placed
        )
      );
    }
  });

  update_feasibility_info(*std::ranges::begin(items));

  auto it = std::ranges::copy(items, m_items.begin()).out;
  std::ranges::for_each(it, m_items.end(), [](Item& item) { item.placed = true; });
}

inline auto State::items() const noexcept -> const Items& {
  return m_items;
}

inline auto State::height_map() const noexcept -> const HeightMap& {
  return m_height_map;
}

inline auto State::feasibility_info() const noexcept -> const FeasibilityInfo& {
  return m_feasibility_info;
}

}  // namespace mcts