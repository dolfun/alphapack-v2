#pragma once
#include <core/memory/alloc_fn_utils.h>
#include <core/memory/memory_block_pool.h>
#include <core/memory/raw_buffer.h>
#include <cuda_runtime.h>

#include <cassert>
#include <new>

namespace alpack {

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
  free_fn(ptr);  // can return error
}

constexpr inline Allocator cuda_host_alloc_free_pair{
  .allocate = detail::cuda_alloc_wrapper<cudaMallocHost>,
  .free = detail::cuda_free_wrapper<cudaFreeHost>
};

constexpr inline Allocator cuda_alloc_free_pair{
  .allocate = detail::cuda_alloc_wrapper<cudaMalloc>,
  .free = detail::cuda_free_wrapper<cudaFree>
};

}  // namespace detail

using PinnedRawBuffer = RawBuffer<detail::cuda_host_alloc_free_pair>;
using CudaRawBuffer = RawBuffer<detail::cuda_alloc_free_pair>;

template <size_t alignment>
using PinnedMemoryPool = MemoryBlockPool<detail::cuda_host_alloc_free_pair, alignment>;

template <size_t alignment>
using CudaMemoryPool = MemoryBlockPool<detail::cuda_alloc_free_pair, alignment>;

};  // namespace alpack
