#pragma once
#include <cstdint>

namespace alpack {

template <typename T>
struct Vec3 {
  T x, y, z;
};

struct Item {
  using shape_type = std::uint8_t;

  Vec3<shape_type> shape;
  bool placed;

  [[nodiscard]] constexpr auto volume() const noexcept -> std::uint32_t {
    return static_cast<std::uint32_t>(shape.x) * static_cast<std::uint32_t>(shape.y) *
           static_cast<std::uint32_t>(shape.z);
  }

  static constexpr Item make_item(shape_type x, shape_type y, shape_type z) {
    return Item{.shape = {x, y, z}, .placed = false};
  }
};

}  // namespace alpack