// cppcheck-suppress-file functionStatic
#pragma once
#include <core/memory/object_pool.h>

namespace alpack {

struct BatchedArrayPoolConfig {
  std::size_t batch_size;
  std::size_t batch_pool_size;
};

template <typename T, std::size_t ArraySize, Allocator Alloc>
class BatchedArrayPool {
public:
  explicit BatchedArrayPool(BatchedArrayPoolConfig config)
      : m_batch_size{ArraySize * config.batch_size},
        m_pool{m_batch_size * config.batch_pool_size} {}

  [[nodiscard]] auto item(this auto& self, std::size_t item_idx) noexcept {
    return self.m_pool.template span<ArraySize>(ArraySize * item_idx);
  }

  [[nodiscard]] auto batch(this auto& self, std::size_t batch_idx) noexcept {
    return self.m_pool.span(self.m_batch_size * batch_idx, self.m_batch_size);
  }

  // cppcheck-suppress constParameterReference
  [[nodiscard]] auto pool(this auto& self) noexcept -> auto& {
    return self.m_pool;
  }

private:
  std::size_t m_batch_size;
  ObjectPool<T, Alloc> m_pool;
};

}  // namespace alpack
