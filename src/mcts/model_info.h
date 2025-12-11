#pragma once

namespace mcts {

struct ModelInfo {
  static constexpr size_t input_feature_count = 2;
  static constexpr size_t additional_input_count = 64;
  static constexpr size_t value_support_count = 101;
};

}  // namespace mcts