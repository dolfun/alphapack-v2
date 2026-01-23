add_library(
  inference allocator.h inference_engine.cc inference_engine.h
            inference_model.cc inference_model.h
)
target_link_torch(inference)
target_link_libraries(inference PUBLIC memory)
enable_project_options(inference)
