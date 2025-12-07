cmake_minimum_required(VERSION 3.20)
project(alphapack)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(
  Python
  COMPONENTS Interpreter
  QUIET
)

if(Python_FOUND)
  message(STATUS "Found Python: ${Python_EXECUTABLE}")

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PYTHON_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
  )

  if(TORCH_PYTHON_PREFIX)
    message(STATUS "Auto-detected PyTorch path: ${TORCH_PYTHON_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_PYTHON_PREFIX}")

  else()
    message(
      WARNING "'torch' installation not found. Relying on CMAKE_PREFIX_PATH."
    )
  endif()

else()
  message(WARNING "Python interpreter not found. Relying on CMAKE_PREFIX_PATH.")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(src)
