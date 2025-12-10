add_library(
  torch_utils memory_pool.h inference_model.h inference_model.cc
              inference_engine.h inference_engine.cc
)
target_compile_features(torch_utils PUBLIC cxx_std_23)
target_include_directories(torch_utils PUBLIC .)
target_link_pytorch(torch_utils)
