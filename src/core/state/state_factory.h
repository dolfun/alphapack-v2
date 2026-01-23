#pragma once
#include <core/state/state.h>

namespace alpack {

struct RandomSequenceStateInfo {
  std::uint64_t seed = 0;
  std::size_t count = 0;
  std::uint32_t min_length = 0;
  std::uint32_t max_length = 0;
  std::uint32_t min_height = 0;
  std::uint32_t max_height = 0;
};

[[nodiscard]] auto make_random_sequence_state(const RandomSequenceStateInfo& info) -> State;

}  // namespace alpack
