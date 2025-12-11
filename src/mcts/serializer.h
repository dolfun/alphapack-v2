#pragma once
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "state.h"

namespace mcts {

template <typename T>
class Serializer;

template <>
class Serializer<State> {
public:
  static auto serialize(const State& state) -> std::string {
    std::pair<const void*, size_t> buffer_infos[3] = {
      {state.m_items.data(), sizeof(Item) * state.m_items.size()},
      {state.m_height_map.data(), sizeof(int8_t) * state.m_height_map.nbytes},
      {state.m_feasibility_info.data(), sizeof(int8_t) * state.m_feasibility_info.nbytes}
    };

    size_t total_size = 0;
    for (auto [_, size] : buffer_infos) {
      total_size += size;
    }

    size_t offset = 0;
    std::string bytes(total_size, ' ');
    for (auto [src, size] : buffer_infos) {
      std::memcpy(bytes.data() + offset, src, size);
      offset += size;
    }

    return bytes;
  }

  static auto unserialize(std::string_view bytes) -> State {
    State state{};
    std::pair<void*, size_t> buffer_infos[3] = {
      {state.m_items.data(), sizeof(Item) * state.m_items.size()},
      {state.m_height_map.data(), sizeof(int8_t) * state.m_height_map.nbytes},
      {state.m_feasibility_info.data(), sizeof(int8_t) * state.m_feasibility_info.nbytes}
    };

    size_t offset = 0;
    for (auto [dest, size] : buffer_infos) {
      std::memcpy(dest, bytes.data() + offset, size);
      offset += size;
    }

    return state;
  };
};

};  // namespace mcts