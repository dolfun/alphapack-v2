add_library(mcts model_adapter.h model_adapter.cc)

target_link_libraries(mcts PRIVATE project_options torch_utils)
