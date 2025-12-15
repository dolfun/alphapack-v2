#pragma once
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <new>
#include <utility>

namespace torch_utils {

using AllocateFn = void* (*)(size_t);
using FreeFn = void (*)(void*) noexcept;

namespace detail {

using CudaAllocateFn = cudaError_t (*)(void**, size_t);
using CudaFreeFn = cudaError_t (*)(void*);

template <CudaAllocateFn allocate_fn>
consteval inline auto wrap_cuda_allocate_fn() -> AllocateFn {
  return +[](size_t size) -> void* {
    void* ptr = nullptr;
    auto error = allocate_fn(&ptr, size);
    if (error != cudaSuccess) {
      throw std::bad_alloc();
    }
    return ptr;
  };
};

template <CudaFreeFn free_fn>
consteval inline auto wrap_cuda_free_fn() -> FreeFn {
  return +[](void* ptr) noexcept {
    auto error = free_fn(ptr);
    assert(error == cudaSuccess);
  };
}

inline auto global_new_fn(size_t size) -> void* {
  return ::operator new(size);
}

inline auto global_delete_fn(void* ptr) noexcept -> void {
  ::operator delete(ptr);
}

constexpr inline auto cuda_malloc_host_fn = wrap_cuda_allocate_fn<cudaMallocHost>();
constexpr inline auto cuda_free_host_fn = wrap_cuda_free_fn<cudaFreeHost>();

constexpr inline auto cuda_malloc_fn = wrap_cuda_allocate_fn<cudaMalloc>();
constexpr inline auto cuda_free_fn = wrap_cuda_free_fn<cudaFree>();

}  // namespace detail

template <AllocateFn allocate_fn, FreeFn free_fn>
class RawBuffer {
public:
  explicit RawBuffer(size_t pool_size) : m_pool_size{pool_size}, m_ptr{nullptr} {
    if (pool_size > 0) {
      m_ptr = allocate_fn(pool_size);
    }
  }

  ~RawBuffer() noexcept {
    if (m_ptr) {
      free_fn(m_ptr);
    }
  }

  RawBuffer(const RawBuffer&) = delete;
  RawBuffer& operator=(const RawBuffer&) = delete;

  RawBuffer(RawBuffer&& other) noexcept
      : m_pool_size{std::exchange(other.m_pool_size, 0)},
        m_ptr{std::exchange(other.m_ptr, nullptr)} {}

  RawBuffer& operator=(RawBuffer&& other) noexcept {
    if (this != &other) {
      if (m_ptr) {
        free_fn(m_ptr);
      }
      m_pool_size = std::exchange(other.m_pool_size, 0);
      m_ptr = std::exchange(other.m_ptr, nullptr);
    }

    return *this;
  }

  void swap(RawBuffer& other) noexcept {
    using std::swap;
    swap(m_pool_size, other.m_pool_size);
    swap(m_ptr, other.m_ptr);
  }

  friend void swap(RawBuffer& a, RawBuffer& b) noexcept {
    a.swap(b);
  }

  [[nodiscard]] auto ptr() noexcept -> void* {
    return m_ptr;
  }

  [[nodiscard]] auto ptr() const noexcept -> const void* {
    return m_ptr;
  }

  [[nodiscard]] auto size() const noexcept -> size_t {
    return m_pool_size;
  }

private:
  size_t m_pool_size;
  void* m_ptr;
};

using DefaultRawBuffer = RawBuffer<detail::global_new_fn, detail::global_delete_fn>;
using PinnedRawBuffer = RawBuffer<detail::cuda_malloc_host_fn, detail::cuda_free_host_fn>;
using CudaRawBuffer = RawBuffer<detail::cuda_malloc_fn, detail::cuda_free_fn>;

}  // namespace torch_utils