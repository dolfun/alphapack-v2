add_subdirectory(mcts)
add_subdirectory(tests)

add_executable(main main.cc)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_compile_features(main PUBLIC cxx_std_23)

if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(
    TARGET main
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
            $<TARGET_FILE_DIR:main>
  )
endif(MSVC)
