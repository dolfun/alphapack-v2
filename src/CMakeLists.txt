add_subdirectory(mcts)
add_subdirectory(bindings)
add_subdirectory(tests)

add_executable(main main.cc)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_compile_features(main PUBLIC cxx_std_23)
target_copy_pytorch_dll(main)
