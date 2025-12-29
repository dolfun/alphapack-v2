#pragma once
#include <cstdint>

namespace alpack {

template <typename T>
struct Vec3 {
  T x, y, z;
};

struct Item {
  using dim_type = std::uint8_t;

  Vec3<dim_type> shape;
  bool placed;

  [[nodiscard]] constexpr auto volume() const noexcept -> std::uint32_t {
    return std::uint32_t{1} * shape.x * shape.y * shape.z;
  }

  static constexpr Item make_item(dim_type x, dim_type y, dim_type z) {
    return Item{.shape = {x, y, z}, .placed = false};
  }
};

}  // namespace alpack