#pragma once
#include <core/memory/allocator.h>
#include <cuda_runtime.h>

#include <cassert>
#include <new>

namespace alpack {

namespace detail {

using CudaAllocateFn = cudaError_t (*)(void**, std::size_t);
using CudaFreeFn = cudaError_t (*)(void*);

template <CudaAllocateFn CudaAllocate, CudaFreeFn CudaFree>
struct CudaAllocator {
  static auto allocate(std::size_t size, [[maybe_unused]] std::align_val_t alignment) -> void* {
    void* ptr = nullptr;  // NOLINT(misc-const-correctness)
    const auto error = CudaAllocate(&ptr, size);
    if (error != cudaSuccess) {
      throw std::bad_alloc();
    }
    return ptr;
  }

  static auto free(void* ptr, [[maybe_unused]] std::align_val_t alignment) noexcept -> void {
    CudaFree(ptr);  // may fail
  }
};

}  // namespace detail

using CudaHostAllocator = detail::CudaAllocator<cudaMallocHost, cudaFreeHost>;
using CudaDeviceAllocator = detail::CudaAllocator<cudaMalloc, cudaFree>;

static_assert(Allocator<CudaHostAllocator>);
static_assert(Allocator<CudaDeviceAllocator>);

};  // namespace alpack
