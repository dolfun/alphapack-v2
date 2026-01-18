add_library(mcts model_adapter.cc model_adapter.h)

target_link_libraries(mcts PRIVATE project_options torch_utils)
