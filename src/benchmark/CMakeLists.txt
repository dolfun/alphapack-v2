add_library(inference_engine_bm inference_engine_bm.cc inference_engine_bm.h)
target_link_libraries(
  inference_engine_bm
  PRIVATE project_options
          state
          inference
          mcts
)
