add_library(
  inference
  allocator.h
  inference_model.h
  inference_model.cc
  inference_engine.h
  inference_engine.cc
)

target_link_torch(inference)
target_link_libraries(
  inference
  PUBLIC memory
  PRIVATE project_options
)
