add_executable(bm_inference_engine bm_inference_engine.cc)
target_compile_features(bm_inference_engine PRIVATE cxx_std_23)
target_link_libraries(bm_inference_engine PRIVATE mcts torch_utils)
target_link_pytorch(bm_inference_engine)

add_executable(unit_tests test_ndarray.cc test_state.cc)
target_link_libraries(unit_tests PRIVATE mcts Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
