#pragma once
#include <core/state/item.h>
#include <core/state/ndarray.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <format>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace alpack {

template <typename T>
class Serializer;

template <typename R, typename T>
concept RangeOf = std::ranges::input_range<R> && std::convertible_to<std::ranges::range_value_t<R>, T>;

class State {
public:
  static constexpr std::size_t max_item_count = 64;
  static constexpr std::size_t bin_length = 10;
  static constexpr std::size_t bin_height = 10;
  static constexpr std::size_t bin_base_size = bin_length * bin_length;
  static constexpr std::size_t action_count = bin_length * bin_length;
  static constexpr std::size_t invalid_feasible_height = std::numeric_limits<std::uint8_t>::max();

  static_assert(bin_length < std::numeric_limits<std::uint8_t>::max());
  static_assert(2 * bin_height < std::numeric_limits<std::uint8_t>::max());

  template <typename T>
  using Array2D = NdArray<T, bin_length, bin_length>;

  using Items = std::array<Item, max_item_count>;
  using HeightMap = Array2D<std::uint8_t>;
  using FeasibilityInfo = Array2D<std::uint8_t>;
  using FeasibilityMask = Array2D<bool>;

  using Action = std::uint8_t;
  static_assert(action_count < std::numeric_limits<Action>::max());

  template <RangeOf<Item> ItemRange>
  explicit State(const ItemRange&);

  [[nodiscard]] auto items() const noexcept -> const Items&;
  [[nodiscard]] auto height_map() const noexcept -> const HeightMap&;
  [[nodiscard]] auto feasibility_info() const noexcept -> const FeasibilityInfo&;

  [[nodiscard]] auto feasibility_mask() const noexcept -> FeasibilityMask;
  [[nodiscard]] auto packing_efficiency() const noexcept -> float;
  [[nodiscard]] auto feasible_actions() const noexcept -> std::vector<Action>;

  [[nodiscard]] auto transition(Action) -> float;

private:
  State() = default;
  friend class Serializer<State>;

  auto update_feasibility_info_with_front_item() noexcept -> void;

  Items m_items{};
  HeightMap m_height_map{};
  FeasibilityInfo m_feasibility_info{};
};

template <RangeOf<Item> ItemRange>
State::State(const ItemRange& items) {
  if (std::ranges::empty(items) || std::ranges::size(items) > max_item_count) {
    throw std::invalid_argument("Invalid number of items in constructor");
  }

  std::ranges::for_each(items, [](const Item& item) {
    if (item.volume() == 0 || item.shape.x > bin_length || item.shape.y > bin_length || item.shape.z > bin_height ||
        item.placed) {
      throw std::invalid_argument(
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

  auto it = std::ranges::copy(items, m_items.begin()).out;
  std::ranges::for_each(it, m_items.end(), [](Item& item) { item.placed = true; });

  update_feasibility_info_with_front_item();
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

}  // namespace alpack