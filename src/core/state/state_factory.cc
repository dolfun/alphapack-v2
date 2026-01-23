#include "state_factory.h"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace alpack {

auto make_random_sequence_state(const RandomSequenceStateInfo& info) -> State {
  if (info.min_length == 0 || info.min_length > info.max_length ||
      info.max_length > State::bin_length) {
    throw std::invalid_argument("Invalid length limits");
  }

  if (info.min_height == 0 || info.min_height > info.max_height ||
      info.max_height > State::bin_height) {
    throw std::invalid_argument("Invalid height limits");
  }

  std::mt19937_64 engine{info.seed};
  std::uniform_int_distribution length_dist{info.min_length, info.max_length};
  std::uniform_int_distribution height_dist{info.min_height, info.max_height};

  std::vector<Item> items(info.count);
  std::ranges::generate(items, [&] {
    const auto l = static_cast<Item::dim_type>(length_dist(engine));
    const auto w = static_cast<Item::dim_type>(length_dist(engine));
    const auto h = static_cast<Item::dim_type>(height_dist(engine));
    return Item::make_item(l, w, h);
  });

  return State{items};
}

}  // namespace alpack
