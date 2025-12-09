add_library(mcts ndarray.h item.h state.h state.cc serializer.h)
target_compile_features(mcts PUBLIC cxx_std_23)
target_include_directories(mcts PUBLIC .)

if(MSVC)
  target_compile_options(mcts PRIVATE /W4 /WX)
else()
  target_compile_options(mcts PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(MSVC)
    message("Enabling optimizations for MSVC")
    target_compile_options(mcts PUBLIC /fp:fast /GL)

  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    message("Enabling optimizations for GCC")
    target_compile_options(mcts PUBLIC -O3 -ffast-math)

  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    message("Enabling optimizations for Clang")
    target_compile_options(mcts PUBLIC -O3 -ffast-math)
  endif()
endif()
