#include <core/state/ndarray.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <type_traits>

using namespace alpack;

TEST_CASE("NdArray: Compile-time properties", "[NdArray][Static]") {
  SECTION("1D Array properties") {
    using Array1D = NdArray<int, 5>;
    STATIC_REQUIRE(Array1D::ndim == 1);
    STATIC_REQUIRE(Array1D::size == 5);
    STATIC_REQUIRE(Array1D::nbytes == 5 * sizeof(int));
    STATIC_REQUIRE(Array1D::shape[0] == 5);
    STATIC_REQUIRE(Array1D::strides[0] == 1);
  }

  SECTION("3D Array properties") {
    using Array3D = NdArray<float, 2, 3, 4>;
    STATIC_REQUIRE(Array3D::ndim == 3);
    STATIC_REQUIRE(Array3D::size == 24);
    STATIC_REQUIRE(Array3D::shape[0] == 2);
    STATIC_REQUIRE(Array3D::shape[1] == 3);
    STATIC_REQUIRE(Array3D::shape[2] == 4);
    STATIC_REQUIRE(Array3D::strides[0] == 12);
    STATIC_REQUIRE(Array3D::strides[1] == 4);
    STATIC_REQUIRE(Array3D::strides[2] == 1);
  }
}

TEST_CASE("NdArray: Construction and Factory", "[NdArray][Construction]") {
  SECTION("Default construction initializes to zero") {
    constexpr NdArray<int, 2, 2> arr{};
    for (auto val : arr) {
      REQUIRE(val == 0);
    }
  }

  SECTION("Factory make_ndarray initialization") {
    constexpr auto arr = make_ndarray<int, 2, 2>(42);
    for (auto val : arr) {
      REQUIRE(val == 42);
    }
  }

  SECTION("Manual Fill") {
    NdArray<int, 2, 2> arr{};
    arr.fill(10);
    for (auto val : arr) {
      REQUIRE(val == 10);
    }
  }

  SECTION("Constexpr construction capability") {
    constexpr auto sum = []() {
      auto a = make_ndarray<int, 2, 2>(10);
      return a[0, 0] + a[1, 1];
    }();
    STATIC_REQUIRE(sum == 20);
  }
}

TEST_CASE("NdArray: Element Access (C++23 operator[])", "[NdArray][Access]") {
  NdArray<int, 3, 3> grid{};
  int counter = 0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      grid[i, j] = counter++;
    }
  }

  SECTION("Read access returns correct values") {
    REQUIRE(grid[0, 0] == 0);
    REQUIRE(grid[1, 1] == 4);
    REQUIRE(grid[2, 2] == 8);
  }

  SECTION("Write access modifies data") {
    grid[1, 1] = 100;
    REQUIRE(grid[1, 1] == 100);
  }
}

TEST_CASE("NdArray: Deducing this correctness", "[NdArray][ConstCorrectness]") {
  auto mut_mat = make_ndarray<int, 2, 2>(1);

  SECTION("Non-const lvalue reference returns non-const reference") {
    mut_mat[0, 0] = 5;
    static_assert(std::is_same_v<decltype(mut_mat[0, 0]), int&>);
    REQUIRE(mut_mat[0, 0] == 5);
  }

  const auto const_mat = make_ndarray<int, 2, 2>(2);
  SECTION("Const lvalue reference returns const reference") {
    static_assert(std::is_same_v<decltype(const_mat[0, 0]), const int&>);
    REQUIRE(const_mat[0, 0] == 2);
  }

  SECTION("R-value access interaction") {
    auto val = make_ndarray<int, 2, 2>(99)[0, 0];
    REQUIRE(val == 99);
  }
}

TEST_CASE("NdArray: Strides and Layout", "[NdArray][Memory]") {
  NdArray<int, 2, 3, 4> tensor{};
  std::iota(tensor.begin(), tensor.end(), 0);

  SECTION("Verify Row-Major Layout via Indexing") {
    REQUIRE(tensor[0, 0, 0] == 0);
    REQUIRE(tensor[0, 0, 1] == 1);
    REQUIRE(tensor[0, 1, 0] == 4);
    REQUIRE(tensor[1, 0, 0] == 12);
  }

  SECTION("Direct pointer access") {
    int* raw_ptr = tensor.data();
    REQUIRE(*raw_ptr == 0);
    REQUIRE(*(raw_ptr + 12) == 12);
  }
}

TEST_CASE("NdArray: Iterators", "[NdArray][Iterators]") {
  NdArray<int, 2, 2> arr{};

  SECTION("Standard algorithms compatibility") {
    std::fill(arr.begin(), arr.end(), 7);
    bool all_seven = std::ranges::all_of(arr, [](int i) { return i == 7; });
    REQUIRE(all_seven);
  }

  SECTION("Range-based for loop") {
    int sum = 0;
    std::fill(arr.begin(), arr.end(), 1);

    for (const auto& val : arr) {
      sum += val;
    }
    REQUIRE(sum == 4);
  }
}