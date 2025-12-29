#pragma once
#include <core/memory/object_pool.h>

namespace alpack {

template <typename T, std::size_t array_size, Allocator Alloc>
class BatchedArrayPool {
public:
  BatchedArrayPool(std::size_t batch_size, std::size_t batch_pool_size)
      : m_batch_size{array_size * batch_size}, m_pool{m_batch_size * batch_pool_size} {}

  [[nodiscard]] auto item(this auto& self, std::size_t item_idx) noexcept {
    return self.m_pool.template span<array_size>(array_size * item_idx);
  }

  [[nodiscard]] auto batch(this auto& self, std::size_t batch_idx) noexcept {
    return self.m_pool.span(self.m_batch_size * batch_idx, self.m_batch_size);
  }

  [[nodiscard]] auto pool(this auto& self) noexcept -> auto& {
    return self.m_pool;
  }

private:
  std::size_t m_batch_size;
  ObjectPool<T, Alloc> m_pool;
};

}  // namespace alpack