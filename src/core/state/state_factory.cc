#include "state_factory.h"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace alpack {

auto make_random_sequence_state(
  std::uint64_t seed,
  std::size_t count,
  std::uint32_t min_length,
  std::uint32_t max_length,
  std::uint32_t min_height,
  std::uint32_t max_height
) -> State {
  if (min_length == 0 || min_length > max_length || max_length > State::bin_length) {
    throw std::invalid_argument("Invalid length limits");
  }

  if (min_height == 0 || min_height > max_height || max_height > State::bin_height) {
    throw std::invalid_argument("Invalid height limits");
  }

  std::mt19937_64 engine{seed};
  std::uniform_int_distribution length_dist{min_length, max_length}, height_dist{min_height, max_height};

  std::vector<Item> items(count);
  std::ranges::generate(items, [&] {
    const auto l = static_cast<Item::dim_type>(length_dist(engine));
    const auto w = static_cast<Item::dim_type>(length_dist(engine));
    const auto h = static_cast<Item::dim_type>(height_dist(engine));
    return Item::make_item(l, w, h);
  });

  return State{items};
}

}  // namespace alpack