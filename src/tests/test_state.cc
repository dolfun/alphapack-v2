#include <core/state/item.h>
#include <core/state/serializer.h>
#include <core/state/state.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace alpack;

TEST_CASE("State: Initialization and Validation", "[State][Init]") {
  SECTION("Constructs with valid items") {
    std::vector<Item> items = {Item::make_item(2, 2, 2)};
    REQUIRE_NOTHROW(State(items));
  }

  SECTION("Throws on empty input") {
    std::vector<Item> empty_items;
    REQUIRE_THROWS_AS(State(empty_items), std::invalid_argument);
  }

  SECTION("Throws on too many items") {
    std::vector<Item> too_many(65, Item::make_item(1, 1, 1));
    REQUIRE_THROWS_AS(State(too_many), std::invalid_argument);
  }

  SECTION("Initial state consistency") {
    std::vector<Item> items = {Item::make_item(10, 10, 1)};
    State state(items);

    auto& hm = state.height_map();
    bool all_zero = true;
    for (auto h : hm) {
      if (h != 0) all_zero = false;
    }
    REQUIRE(all_zero);

    REQUIRE(state.packing_efficiency() == 0.0f);
  }
}

TEST_CASE("State: Feasibility Logic", "[State][Feasibility]") {
  SECTION("Item fits everywhere (1x1)") {
    std::vector<Item> items = {Item::make_item(1, 1, 1)};
    State state(items);

    auto actions = state.feasible_actions();
    REQUIRE(actions.size() == 100);
    REQUIRE(state.feasibility_mask()[0, 0] == true);
    REQUIRE(state.feasibility_mask()[9, 9] == true);
  }

  SECTION("Item fits exactly (10x10)") {
    std::vector<Item> items = {Item::make_item(10, 10, 1)};
    State state(items);

    auto actions = state.feasible_actions();
    REQUIRE(actions.size() == 1);
    REQUIRE(actions[0] == 0);

    REQUIRE(state.feasibility_mask()[0, 0] == true);
    REQUIRE(state.feasibility_mask()[0, 1] == false);
  }
}

TEST_CASE("State: Transitions and Height Map Updates", "[State][Transition]") {
  std::vector<Item> items = {Item::make_item(5, 5, 5), Item::make_item(5, 5, 5)};
  State state(items);

  SECTION("Placing first item updates height map") {
    (void)state.transition(0);

    auto& hm = state.height_map();
    REQUIRE(hm[0, 0] == 5);
    REQUIRE(hm[4, 4] == 5);
    REQUIRE(hm[0, 5] == 0);
    REQUIRE(hm[5, 0] == 0);

    auto mask = state.feasibility_mask();
    REQUIRE(mask[0, 0] == true);
    REQUIRE(state.feasibility_info()[0, 0] == 5);
  }

  SECTION("Stacking items to max height") {
    (void)state.transition(0);
    (void)state.transition(0);

    auto& hm = state.height_map();
    REQUIRE(hm[0, 0] == 10);
  }
}

TEST_CASE("State: Packing Efficiency", "[State][Efficiency]") {
  const std::vector<Item> items = {Item::make_item(10, 10, 5), Item::make_item(1, 1, 1)};
  State state(items);

  REQUIRE(state.packing_efficiency() == 0.0f);
  (void)state.transition(0);
  REQUIRE(state.packing_efficiency() == 0.5f);
}

TEST_CASE("State: Item Rotation and Queue Management", "[State][Queue]") {
  const std::vector<Item> items = {Item::make_item(10, 10, 1), Item::make_item(1, 1, 1)};
  State state(items);

  REQUIRE(state.feasible_actions().size() == 1);
  (void)state.transition(0);
  REQUIRE(state.feasible_actions().size() == 100);
  (void)state.transition(99);
  REQUIRE(state.feasible_actions().empty());
}

TEST_CASE("State: Impossible Stacking", "[State][Constraints]") {
  const std::vector<Item> items = {Item::make_item(5, 5, 6), Item::make_item(5, 5, 5)};
  State state(items);
  (void)state.transition(0);
  auto actions = state.feasible_actions();

  const bool can_stack_at_0 = std::ranges::find(actions, 0) != actions.end();
  REQUIRE_FALSE(can_stack_at_0);

  const bool can_place_at_side = std::ranges::find(actions, 5) != actions.end();
  REQUIRE(can_place_at_side);
}

TEST_CASE("State: Advanced Input Validation", "[State][Validation]") {
  SECTION("Throws on zero dimensions") {
    const std::vector<Item> items = {Item::make_item(0, 5, 5)};
    REQUIRE_THROWS_AS(State(items), std::invalid_argument);
  }

  SECTION("Throws on dimensions exceeding bin size") {
    const std::vector<Item> items = {Item::make_item(11, 1, 1)};
    REQUIRE_THROWS_AS(State(items), std::invalid_argument);
  }
}

TEST_CASE("State: Complex Geometry and Gravity", "[State][Feasibility]") {
  std::vector<Item> items = {Item::make_item(1, 1, 5), Item::make_item(2, 1, 1)};
  State state(items);

  (void)state.transition(0);
  auto& hm = state.height_map();
  REQUIRE(hm[0, 0] == 5);
  REQUIRE(hm[1, 0] == 0);

  auto info = state.feasibility_info();
  REQUIRE(info[0, 0] == 5);

  (void)state.transition(0);
  REQUIRE(hm[0, 0] == 6);
  REQUIRE(hm[1, 0] == 6);
}

TEST_CASE("State: Multi-Cell Boundary Limits", "[State][Feasibility]") {
  std::vector<Item> items = {Item::make_item(2, 2, 1)};
  State state(items);

  auto mask = state.feasibility_mask();

  REQUIRE(mask[0, 0] == true);
  REQUIRE(mask[8, 8] == true);
  REQUIRE(mask[9, 0] == false);
  REQUIRE(mask[9, 8] == false);
  REQUIRE(mask[0, 9] == false);
  REQUIRE(mask[8, 9] == false);
  REQUIRE(mask[9, 9] == false);

  auto actions = state.feasible_actions();
  bool has_edge_action = false;
  for (auto a : actions) {
    if (a % 10 == 9) has_edge_action = true;
    if (a >= 90) has_edge_action = true;
  }
  REQUIRE_FALSE(has_edge_action);
}

TEST_CASE("State: Reward Calculation", "[State][Reward]") {
  const std::vector<Item> items = {Item::make_item(1, 1, 1), Item::make_item(1, 1, 1)};
  State state(items);

  constexpr float scaling = 64.0f * 65.0f / 2.0f;

  const float r1 = state.transition(0);
  REQUIRE(r1 == Catch::Approx(1.0f / scaling));

  const float r2 = state.transition(0);
  REQUIRE(r2 == Catch::Approx(2.0f / scaling));
}

TEST_CASE("State: Copy Independence", "[State][Copy]") {
  std::vector<Item> items = {Item::make_item(5, 5, 5), Item::make_item(5, 5, 5)};
  State s1(items);
  (void)s1.transition(0);

  State s2 = s1;
  REQUIRE(s2.height_map()[0, 0] == 5);

  (void)s2.transition(0);
  REQUIRE(s2.height_map()[0, 0] == 10);
  REQUIRE(s1.height_map()[0, 0] == 5);
  REQUIRE(s1.packing_efficiency() != s2.packing_efficiency());
}

TEST_CASE("State: Serialization Round-Trip", "[State][Serializer]") {
  SECTION("Serialize and restore initial state") {
    std::vector<Item> items = {Item::make_item(2, 2, 2), Item::make_item(1, 1, 1)};
    State original_state(items);

    std::string bytes = Serializer<State>::serialize(original_state);
    REQUIRE(bytes.size() > 0);

    State restored_state = Serializer<State>::unserialize(bytes);
    bool hm_match = true;
    auto& hm_orig = original_state.height_map();
    auto& hm_rest = restored_state.height_map();
    for (int i = 0; i < 100; ++i) {
      if (hm_orig[i / 10, i % 10] != hm_rest[i / 10, i % 10]) hm_match = false;
    }
    REQUIRE(hm_match);
    REQUIRE(restored_state.packing_efficiency() == original_state.packing_efficiency());
  }

  SECTION("Serialize and restore partially packed state") {
    std::vector<Item> items = {Item::make_item(5, 5, 5), Item::make_item(2, 2, 1)};
    State original_state(items);

    (void)original_state.transition(0);

    REQUIRE(original_state.packing_efficiency() > 0.0f);
    auto& hm_orig = original_state.height_map();
    REQUIRE(hm_orig[0, 0] == 5);
    std::string bytes = Serializer<State>::serialize(original_state);

    State restored_state = Serializer<State>::unserialize(bytes);
    auto& hm_rest = restored_state.height_map();
    REQUIRE(hm_rest[0, 0] == 5);
    REQUIRE(hm_rest[4, 4] == 5);
    REQUIRE(hm_rest[5, 5] == 0);

    auto mask_orig = original_state.feasibility_mask();
    auto mask_rest = restored_state.feasibility_mask();

    bool mask_match = true;
    if (mask_orig[0, 0] != mask_rest[0, 0]) mask_match = false;
    if (mask_orig[9, 9] != mask_rest[9, 9]) mask_match = false;
    REQUIRE(mask_match);

    auto actions_orig = original_state.feasible_actions();
    auto actions_rest = restored_state.feasible_actions();

    REQUIRE(actions_rest.size() == actions_orig.size());
    if (!actions_orig.empty()) {
      REQUIRE(actions_rest[0] == actions_orig[0]);
      REQUIRE(actions_rest.back() == actions_orig.back());
    }

    (void)restored_state.transition(actions_rest[0]);
    REQUIRE(restored_state.height_map()[0, 0] == 6);
  }

  SECTION("Byte-level consistency") {
    std::vector<Item> items = {Item::make_item(1, 1, 1)};
    State s1(items);
    State s2(items);

    std::string b1 = Serializer<State>::serialize(s1);
    std::string b2 = Serializer<State>::serialize(s2);
    REQUIRE(b1 == b2);

    (void)s1.transition(0);
    std::string b1_mod = Serializer<State>::serialize(s1);

    REQUIRE(b1 != b1_mod);
    REQUIRE(b1.size() == b1_mod.size());
  }
}