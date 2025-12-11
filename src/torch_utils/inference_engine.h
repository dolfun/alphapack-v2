#pragma once

#include <memory>

#include "inference_model.h"

namespace torch_utils {

class InferenceResult {
public:
  InferenceResult();
  ~InferenceResult();

  InferenceResult(const InferenceResult&) = delete;
  InferenceResult& operator=(const InferenceResult&) = delete;
  InferenceResult(InferenceResult&&) noexcept;
  InferenceResult& operator=(InferenceResult&&) noexcept;

  [[nodiscard]] auto is_done() const noexcept -> bool;

private:
  friend class InferenceEngine;
  struct Impl;
  std::unique_ptr<Impl> m_pimpl;
};

class InferenceEngine {
public:
  InferenceEngine(InferenceModel model, size_t pool_size);
  ~InferenceEngine();

  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;
  InferenceEngine(InferenceEngine&&) = delete;
  InferenceEngine& operator=(InferenceEngine&&) = delete;

  [[nodiscard]] auto run(const InferenceInfo&) -> InferenceResult;

private:
  struct Impl;
  std::unique_ptr<Impl> m_pimpl;
};

}  // namespace torch_utils