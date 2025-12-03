#pragma once
#include <array>
#include <type_traits>

namespace mcts {

template <typename T, size_t... Dimensions>
  requires(std::is_arithmetic_v<T> && sizeof...(Dimensions) > 0 && ((Dimensions > 0) && ...))
class NdArray {
public:
  static constexpr size_t ndims = sizeof...(Dimensions);

  static constexpr std::array dims { Dimensions... };

  static constexpr auto strides = [] {
    std::array<size_t, ndims> strides {};

    strides[ndims - 1] = 1;
    for (size_t i = ndims - 1; i > 0; --i) {
      strides[i - 1] = strides[i] * dims[i];
    }

    return strides;
  }();

  static constexpr size_t size = (Dimensions * ...);
  static constexpr size_t bytes = sizeof(T) * size;

  constexpr NdArray() = default;

  constexpr explicit NdArray(T val) noexcept {
    m_data.fill(val);
  }

  template <typename... Indices>
    requires(sizeof...(Indices) == ndims && (std::is_integral_v<Indices> && ...))
  constexpr inline auto operator[](this auto&& self, Indices... indices) noexcept
    -> decltype(auto) {
    size_t idx = 0, dim_idx = 0;
    ((idx += indices * strides[dim_idx++]), ...);
    return self.m_data[idx];
  }

  [[nodiscard]] constexpr inline auto data(this auto& self) noexcept {
    return self.m_data.data();
  }

private:
  std::array<T, size> m_data;
};

}  // namespace mcts