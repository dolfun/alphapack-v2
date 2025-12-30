add_executable(inference_engine_bm inference_engine_bm.cc)
target_link_libraries(
        inference_engine_bm
        PRIVATE project_options
        state
        inference
        mcts
)
target_copy_torch_dll(inference_engine_bm)
