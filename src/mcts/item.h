#pragma once
#include <cstdint>

namespace mcts {

template <typename T>
struct Vec3 {
  T x, y, z;
};

struct Item {
  Vec3<uint8_t> shape;
  bool placed;

  [[nodiscard]] constexpr auto volume() const noexcept -> uint32_t {
    return static_cast<uint32_t>(shape.x) * static_cast<uint32_t>(shape.y) *
           static_cast<uint32_t>(shape.z);
  }

  static constexpr Item make_item(uint8_t x, uint8_t y, uint8_t z) {
    return Item{.shape = {x, y, z}, .placed = false};
  }
};

}  // namespace mcts