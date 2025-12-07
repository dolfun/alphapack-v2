#pragma once
#include <array>
#include <type_traits>
#include <utility>

namespace mcts {

template <typename T, size_t... Dimensions>
  requires(std::is_arithmetic_v<T> && sizeof...(Dimensions) > 0 && ((Dimensions > 0) && ...))
class NdArray {
public:
  static constexpr size_t ndims = sizeof...(Dimensions);
  static constexpr std::array dims = { Dimensions... };
  static constexpr size_t size = (Dimensions * ...);
  static constexpr size_t bytes = sizeof(T) * size;

  constexpr NdArray() = default;

  template <typename... Indices>
    requires(sizeof...(Indices) == ndims && (std::is_integral_v<Indices> && ...))
  [[nodiscard]] constexpr auto operator[](this auto&& self, Indices... indices) noexcept
    -> decltype(auto) {
    const auto offset = compute_offset(std::make_index_sequence<ndims>{}, indices...);
    return self.m_data[offset];
  }

  constexpr void fill(T val) noexcept {
    m_data.fill(val);
  }

  [[nodiscard]] constexpr auto data(this auto& self) noexcept {
    return self.m_data.data();
  }

  [[nodiscard]] constexpr auto begin(this auto& self) noexcept {
    return self.m_data.begin();
  }

  [[nodiscard]] constexpr auto end(this auto& self) noexcept {
    return self.m_data.end();
  }

private:
  std::array<T, size> m_data{};

  static constexpr auto strides = [] consteval {
    std::array<size_t, ndims> strides{};
    strides[ndims - 1] = 1;
    for (auto i = ndims - 1; i > 0; --i) {
      strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
  }();

  template <size_t... Is>
  static constexpr auto compute_offset(std::index_sequence<Is...>, auto... indices) -> size_t {
    return ((static_cast<size_t>(indices) * strides[Is]) + ...);
  }
};

template <typename T, size_t... Dimensions>
constexpr inline auto make_ndarray(T val = {}) {
  NdArray<T, Dimensions...> arr{};
  arr.fill(val);
  return arr;
}

}  // namespace mcts