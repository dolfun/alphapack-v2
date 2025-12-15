add_library(
  torch_utils
  memory.h
  inference_model.h
  inference_model.cc
  inference_engine.h
  inference_engine.cc
)

target_link_torch(torch_utils)
target_link_libraries(
  torch_utils
  PUBLIC memory
  PRIVATE project_options
)
