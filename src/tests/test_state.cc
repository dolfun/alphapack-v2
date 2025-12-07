#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "item.h"
#include "state.h"

using namespace mcts;

TEST_CASE("State: Initialization and Validation", "[State][Init]") {
  SECTION("Constructs with valid items") {
    std::vector<Item> items = { Item::make_item(2, 2, 2) };
    REQUIRE_NOTHROW(State(items));
  }

  SECTION("Throws on empty input") {
    std::vector<Item> empty_items;
    REQUIRE_THROWS_AS(State(empty_items), std::runtime_error);
  }

  SECTION("Throws on too many items") {
    std::vector<Item> too_many(65, Item::make_item(1, 1, 1));
    REQUIRE_THROWS_AS(State(too_many), std::runtime_error);
  }

  SECTION("Initial state consistency") {
    std::vector<Item> items = { Item::make_item(10, 10, 1) };
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
    std::vector<Item> items = { Item::make_item(1, 1, 1) };
    State state(items);

    auto actions = state.feasible_actions();
    REQUIRE(actions.size() == 100);
    REQUIRE(state.feasibility_mask()[0, 0] == true);
    REQUIRE(state.feasibility_mask()[9, 9] == true);
  }

  SECTION("Item fits exactly (10x10)") {
    std::vector<Item> items = { Item::make_item(10, 10, 1) };
    State state(items);

    auto actions = state.feasible_actions();
    REQUIRE(actions.size() == 1);
    REQUIRE(actions[0] == 0);

    REQUIRE(state.feasibility_mask()[0, 0] == true);
    REQUIRE(state.feasibility_mask()[0, 1] == false);
  }
}

TEST_CASE("State: Transitions and Height Map Updates", "[State][Transition]") {
  std::vector<Item> items = { Item::make_item(5, 5, 5), Item::make_item(5, 5, 5) };
  State state(items);

  SECTION("Placing first item updates height map") {
    float reward = state.transition(0);

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
  std::vector<Item> items = { Item::make_item(10, 10, 5), Item::make_item(1, 1, 1) };
  State state(items);

  REQUIRE(state.packing_efficiency() == 0.0f);
  (void)state.transition(0);
  REQUIRE(state.packing_efficiency() == 0.5f);
}

TEST_CASE("State: Item Rotation and Queue Management", "[State][Queue]") {
  std::vector<Item> items = { Item::make_item(10, 10, 1), Item::make_item(1, 1, 1) };
  State state(items);

  REQUIRE(state.feasible_actions().size() == 1);
  (void)state.transition(0);
  REQUIRE(state.feasible_actions().size() == 100);
  (void)state.transition(99);
  REQUIRE(state.feasible_actions().empty());
}

TEST_CASE("State: Impossible Stacking", "[State][Constraints]") {
  std::vector<Item> items = { Item::make_item(5, 5, 6), Item::make_item(5, 5, 5) };
  State state(items);
  (void)state.transition(0);
  auto actions = state.feasible_actions();

  bool can_stack_at_0 = std::ranges::find(actions, 0) != actions.end();
  REQUIRE_FALSE(can_stack_at_0);

  bool can_place_at_side = std::ranges::find(actions, 5) != actions.end();
  REQUIRE(can_place_at_side);
}

TEST_CASE("State: Advanced Input Validation", "[State][Validation]") {
  SECTION("Throws on zero dimensions") {
    std::vector<Item> items = { Item::make_item(0, 5, 5) };
    REQUIRE_THROWS_AS(State(items), std::runtime_error);
  }

  SECTION("Throws on dimensions exceeding bin size") {
    std::vector<Item> items = { Item::make_item(11, 1, 1) };
    REQUIRE_THROWS_AS(State(items), std::runtime_error);
  }
}

TEST_CASE("State: Complex Geometry and Gravity", "[State][Feasibility]") {
  std::vector<Item> items = { Item::make_item(1, 1, 5), Item::make_item(2, 1, 1) };
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
  std::vector<Item> items = { Item::make_item(2, 2, 1) };
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
  std::vector<Item> items = { Item::make_item(1, 1, 1), Item::make_item(1, 1, 1) };
  State state(items);

  float scaling = 64.0f * 65.0f / 2.0f;

  float r1 = state.transition(0);
  REQUIRE(r1 == Catch::Approx(1.0f / scaling));

  float r2 = state.transition(0);
  REQUIRE(r2 == Catch::Approx(2.0f / scaling));
}

TEST_CASE("State: Copy Independence", "[State][Copy]") {
  std::vector<Item> items = { Item::make_item(5, 5, 5), Item::make_item(5, 5, 5) };
  State s1(items);
  (void)s1.transition(0);

  State s2 = s1;
  REQUIRE(s2.height_map()[0, 0] == 5);

  (void)s2.transition(0);
  REQUIRE(s2.height_map()[0, 0] == 10);
  REQUIRE(s1.height_map()[0, 0] == 5);
  REQUIRE(s1.packing_efficiency() != s2.packing_efficiency());
}