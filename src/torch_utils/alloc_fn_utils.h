#pragma once
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <new>

namespace torch_utils {

using AllocateFn = void* (*)(size_t, std::align_val_t);
using FreeFn = void (*)(void*, std::align_val_t) noexcept;

template <size_t alignment>
concept Alignment = (alignment > 0 && (alignment & (alignment - 1)) == 0);

namespace detail {

using CudaAllocateFn = cudaError_t (*)(void**, size_t);
using CudaFreeFn = cudaError_t (*)(void*);

template <CudaAllocateFn allocate_fn>
inline auto cuda_alloc_wrapper(size_t size, std::align_val_t) -> void* {
  void* ptr = nullptr;
  auto error = allocate_fn(&ptr, size);
  if (error != cudaSuccess) {
    throw std::bad_alloc();
  }
  return ptr;
}

template <CudaFreeFn free_fn>
inline auto cuda_free_wrapper(void* ptr, std::align_val_t) noexcept -> void {
  auto error = free_fn(ptr);
  assert(error == cudaSuccess);
}

}  // namespace detail

struct Allocator {
  AllocateFn allocate;
  FreeFn free;
};

constexpr inline Allocator global_alloc_free_pair{
  .allocate = (::operator new),
  .free = (::operator delete)
};

constexpr inline Allocator cuda_host_alloc_free_pair{
  .allocate = detail::cuda_alloc_wrapper<cudaMallocHost>,
  .free = detail::cuda_free_wrapper<cudaFreeHost>
};

constexpr inline Allocator cuda_alloc_free_pair{
  .allocate = detail::cuda_alloc_wrapper<cudaMalloc>,
  .free = detail::cuda_free_wrapper<cudaFree>
};

}  // namespace torch_utils