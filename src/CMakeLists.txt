add_subdirectory(mcts)
add_subdirectory(bindings)
add_subdirectory(tests)

add_executable(main main.cc)
target_link_pytorch(main)
target_compile_features(main PUBLIC cxx_std_23)
