#pragma once
#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <new>
#include <utility>

namespace torch_utils {

using AllocatorFn = cudaError_t (*)(void**, size_t);
using FreeFn = cudaError_t (*)(void*);

template <AllocatorFn allocate, FreeFn free>
class MemoryPool {
public:
  MemoryPool(size_t pool_size, size_t alignment = 64)
      : m_pool_size{ pool_size }, m_alignment{ alignment } {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      throw std::bad_alloc();
    }

    cudaError_t err = cudaMallocHost(&m_raw_pool_ptr, m_pool_size + m_alignment);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }

    m_aligned_pool_ptr = get_aligned_ptr(m_raw_pool_ptr, m_alignment);
  }

  ~MemoryPool() noexcept {
    if (m_raw_pool_ptr) {
      cudaFreeHost(m_raw_pool_ptr);
    }
  }

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  MemoryPool(MemoryPool&& other) noexcept
      : m_pool_size{ std::exchange(other.m_pool_size, 0) },
        m_alignment{ std::exchange(other.m_alignment, 0) },
        m_raw_pool_ptr{ std::exchange(other.m_raw_pool_ptr, nullptr) },
        m_aligned_pool_ptr{ std::exchange(other.m_aligned_pool_ptr, nullptr) } {}

  MemoryPool& operator=(MemoryPool&& other) noexcept {
    MemoryPool temp{ std::move(other) };
    swap(temp);
    return *this;
  }

  void swap(MemoryPool& other) noexcept {
    using std::swap;
    swap(m_pool_size, other.m_pool_size);
    swap(m_alignment, other.m_alignment);
    swap(m_raw_pool_ptr, other.m_raw_pool_ptr);
    swap(m_aligned_pool_ptr, other.m_aligned_pool_ptr);
  }

  friend void swap(MemoryPool& a, MemoryPool& b) noexcept {
    a.swap(b);
  }

  [[nodiscard]] auto ptr() noexcept -> void* {
    return m_aligned_pool_ptr;
  }

  [[nodiscard]] auto ptr() const noexcept -> const void* {
    return m_aligned_pool_ptr;
  }

  [[nodiscard]] auto size() const noexcept -> size_t {
    return m_pool_size;
  }

  [[nodiscard]] auto alignment() const noexcept -> size_t {
    return m_alignment;
  }

private:
  static auto get_aligned_ptr(void* ptr, size_t alignment) -> void* {
    auto current_address = std::bit_cast<std::uintptr_t>(ptr);
    auto mask = alignment - 1;
    auto aligned_address = (current_address + mask) & ~mask;
    return std::bit_cast<void*>(aligned_address);
  }

  size_t m_pool_size, m_alignment;
  void* m_raw_pool_ptr = nullptr;
  void* m_aligned_pool_ptr = nullptr;
};

using PinnedMemoryPool = MemoryPool<cudaMallocHost, cudaFreeHost>;
using CudaMemoryPool = MemoryPool<cudaMalloc, cudaFree>;

}  // namespace torch_utils