#pragma once
#include <core/state/state.h>

namespace alpack {

[[nodiscard]] auto make_random_sequence_state(
  std::uint64_t seed,
  std::size_t count,
  std::uint32_t min_length,
  std::uint32_t max_length,
  std::uint32_t min_height,
  std::uint32_t max_height
) -> State;

}  // namespace alpack