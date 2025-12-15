#pragma once
#include <new>

namespace alpack {

using AllocateFn = void* (*)(size_t, std::align_val_t);
using FreeFn = void (*)(void*, std::align_val_t) noexcept;

template <size_t alignment>
concept Alignment = (alignment > 0 && (alignment & (alignment - 1)) == 0);

struct Allocator {
  AllocateFn allocate;
  FreeFn free;
};

constexpr inline Allocator global_alloc_free_pair{.allocate = (::operator new), .free = (::operator delete)};

}  // namespace alpack