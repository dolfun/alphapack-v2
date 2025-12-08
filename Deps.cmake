include(FetchContent)

# Catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

# Python
find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# PyTorch
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PYTHON_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT TORCH_PYTHON_PREFIX)
  message(
    FATAL_ERROR
      "PyTorch installation not found for Python interpreter: ${Python_EXECUTABLE}"
  )
endif()

message(STATUS "Found PyTorch: ${TORCH_PYTHON_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PYTHON_PREFIX}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

function(target_copy_pytorch_dll TARGET_NAME)
  if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(
      TARGET ${TARGET_NAME}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
              $<TARGET_FILE_DIR:${TARGET_NAME}>
    )
  endif(MSVC)
endfunction()

# pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(pybind11)

# stubgen
if(WIN32)
  set(STUBGEN_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/Scripts/stubgen.exe")
else()
  set(STUBGEN_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/stubgen")
endif()

function(py_module_add_stubgen TARGET_NAME)
  if(EXISTS ${STUBGEN_EXECUTABLE})
    set(STUB_OUTPUT "${MODULE_OUTPUT_DIR}/${TARGET_NAME}.pyi")

    add_custom_command(
      OUTPUT ${STUB_OUTPUT}
      COMMAND ${STUBGEN_EXECUTABLE} -m ${TARGET_NAME} -o .
      WORKING_DIRECTORY ${MODULE_OUTPUT_DIR}
      DEPENDS ${TARGET_NAME}
    )

    add_custom_target(${TARGET_NAME}_stubgen ALL DEPENDS ${STUB_OUTPUT})
  endif()
endfunction()
