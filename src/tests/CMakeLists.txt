add_executable(unit_tests test_ndarray.cc test_state.cc)
target_link_libraries(unit_tests PRIVATE mcts Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
