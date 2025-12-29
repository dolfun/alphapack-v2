#pragma once
#include <core/memory/allocator.h>
#include <cuda_runtime.h>

#include <cassert>
#include <new>

namespace alpack {

namespace detail {

using CudaAllocateFn = cudaError_t (*)(void**, std::size_t);
using CudaFreeFn = cudaError_t (*)(void*);

template <CudaAllocateFn cuda_allocate, CudaFreeFn cuda_free>
struct CudaAllocator {
  static auto allocate(std::size_t size, std::align_val_t) -> void* {
    void* ptr = nullptr;
    const auto error = cuda_allocate(&ptr, size);
    if (error != cudaSuccess) {
      throw std::bad_alloc();
    }
    return ptr;
  }

  static auto free(void* ptr, std::align_val_t) noexcept -> void {
    cuda_free(ptr);  // may fail
  }
};

}  // namespace detail

using CudaHostAllocator = detail::CudaAllocator<cudaMallocHost, cudaFreeHost>;
using CudaDeviceAllocator = detail::CudaAllocator<cudaMalloc, cudaFree>;

static_assert(Allocator<CudaHostAllocator>);
static_assert(Allocator<CudaDeviceAllocator>);

};  // namespace alpack
