add_library(
  memory INTERFACE allocator.h storage_buffer.h object_pool.h
  batched_array_pool.h
)
enable_project_options(memory)
