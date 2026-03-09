#pragma once
#include <core/state/state.h>

#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace alpack {

template <typename T>
class Serializer;

template <>
class Serializer<State> {
public:
  static auto serialize(const State& state) -> std::string {
    const std::array<std::pair<const void*, std::size_t>, 3> buffer_infos = {
      {{state.m_items.data(), sizeof(Item) * state.m_items.size()},
       {state.m_height_map.data(), state.m_height_map.nbytes},
       {state.m_feasibility_info.data(), state.m_feasibility_info.nbytes}}
    };

    auto total_size = std::ranges::fold_left(buffer_infos | std::views::values, 0uz, std::plus<>());
    std::size_t offset = 0;
    std::string bytes(total_size, ' ');
    for (auto [src, size] : buffer_infos) {
      std::memcpy(bytes.data() + offset, src, size);
      offset += size;
    }

    return bytes;
  }

  static auto unserialize(std::string_view bytes) -> State {
    State state{};
    const std::array<std::pair<void*, std::size_t>, 3> buffer_infos = {
      {{state.m_items.data(), sizeof(Item) * state.m_items.size()},
       {state.m_height_map.data(), state.m_height_map.nbytes},
       {state.m_feasibility_info.data(), state.m_feasibility_info.nbytes}}
    };

    std::size_t offset = 0;
    for (auto [dest, size] : buffer_infos) {
      std::memcpy(dest, bytes.data() + offset, size);
      offset += size;
    }

    return state;
  };
};

};  // namespace alpack