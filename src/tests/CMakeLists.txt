include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.11.0
)

FetchContent_MakeAvailable(Catch2)

add_executable(unit_tests test_ndarray.cc test_state.cc)
target_link_libraries(unit_tests PRIVATE mcts Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
