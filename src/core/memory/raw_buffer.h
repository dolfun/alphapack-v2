#pragma once
#include <core/memory/alloc_fn_utils.h>

#include <cstdint>
#include <new>
#include <utility>

namespace alpack {

template <Allocator allocator, size_t alignment = alignof(std::max_align_t)>
  requires Alignment<alignment>
class RawBuffer {
public:
  explicit RawBuffer(size_t size) : m_size{size}, m_ptr{nullptr} {
    if (size > 0) {
      m_ptr = do_allocate(size);

      if (reinterpret_cast<std::uintptr_t>(m_ptr) % alignment != 0) {
        do_free(m_ptr);
        throw std::bad_alloc();
      }
    }
  }

  ~RawBuffer() noexcept {
    if (m_ptr) {
      do_free(m_ptr);
    }
  }

  RawBuffer(const RawBuffer&) = delete;
  RawBuffer& operator=(const RawBuffer&) = delete;

  RawBuffer(RawBuffer&& other) noexcept
      : m_size{std::exchange(other.m_size, 0)}, m_ptr{std::exchange(other.m_ptr, nullptr)} {}

  RawBuffer& operator=(RawBuffer&& other) noexcept {
    if (this != &other) {
      if (m_ptr) {
        do_free(m_ptr);
      }
      m_size = std::exchange(other.m_size, 0);
      m_ptr = std::exchange(other.m_ptr, nullptr);
    }

    return *this;
  }

  void swap(RawBuffer& other) noexcept {
    using std::swap;
    swap(m_size, other.m_size);
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
    return m_size;
  }

private:
  static auto do_allocate(size_t size) -> void* {
    return allocator.allocate(size, static_cast<std::align_val_t>(alignment));
  }

  static auto do_free(void* ptr) noexcept -> void {
    allocator.free(ptr, static_cast<std::align_val_t>(alignment));
  }

  size_t m_size;
  void* m_ptr;
};

using DefaultRawBuffer = RawBuffer<global_alloc_free_pair>;

}  // namespace alpack