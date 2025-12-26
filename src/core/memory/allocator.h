#pragma once
#include <concepts>
#include <cstddef>
#include <new>

namespace alpack {

template <std::size_t Alignment>
concept PowerOfTwo = (Alignment > 0 && (Alignment & (Alignment - 1)) == 0);

template <typename Alloc>
concept Allocator = requires(std::size_t size, std::align_val_t align, void* ptr) {
  { Alloc::allocate(size, align) } -> std::same_as<void*>;
  { Alloc::free(ptr, align) } noexcept -> std::same_as<void>;
};

struct DefaultAllocator {
  static auto allocate(std::size_t size, std::align_val_t align) -> void* {
    return ::operator new(size, align);
  }

  static auto free(void* ptr, std::align_val_t align) noexcept -> void {
    ::operator delete(ptr, align);
  }
};

static_assert(Allocator<DefaultAllocator>);

}  // namespace alpack