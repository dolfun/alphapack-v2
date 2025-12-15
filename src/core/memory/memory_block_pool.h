#pragma once
#include <core/memory/alloc_fn_utils.h>
#include <core/memory/raw_buffer.h>

#include <cassert>
#include <type_traits>

namespace alpack {

template <Allocator allocator, size_t alignment>
  requires Alignment<alignment>
class MemoryBlockPool {
public:
  MemoryBlockPool(size_t block_size, size_t block_count)
      : m_block_size{block_size},
        m_block_count{block_count},
        m_block_stride{align_up(block_size)},
        m_storage{m_block_stride * m_block_count} {
    assert(m_block_size > 0 && m_block_count > 0);
  }

  [[nodiscard]] auto operator[](this auto& self, size_t idx) noexcept {
    assert(idx < self.m_block_count);
    auto raw_ptr = self.m_storage.ptr();

    using BytePtr = std::conditional_t<
      std::is_const_v<std::remove_pointer_t<decltype(raw_ptr)>>,
      const std::byte*,
      std::byte*>;

    auto byte_ptr = static_cast<BytePtr>(raw_ptr);
    return static_cast<decltype(raw_ptr)>(byte_ptr + idx * self.m_block_stride);
  }

  [[nodiscard]] auto block_size() const noexcept -> size_t {
    return m_block_size;
  }

  [[nodiscard]] auto block_stride() const noexcept -> size_t {
    return m_block_stride;
  }

  [[nodiscard]] auto block_count() const noexcept -> size_t {
    return m_block_count;
  }

private:
  static constexpr auto align_up(size_t n) noexcept -> size_t {
    return (n + (alignment - 1)) & ~(alignment - 1);
  }

  size_t m_block_size{};
  size_t m_block_count{};
  size_t m_block_stride{};
  RawBuffer<allocator, alignment> m_storage;
};

template <size_t alignment>
using DefaultMemoryPool = MemoryBlockPool<global_alloc_free_pair, alignment>;

}  // namespace alpack