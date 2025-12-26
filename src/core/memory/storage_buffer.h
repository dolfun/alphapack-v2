#pragma once
#include <core/memory/allocator.h>

#include <new>
#include <utility>

namespace alpack {

template <Allocator Alloc = DefaultAllocator, std::size_t Alignment = alignof(std::max_align_t)>
  requires PowerOfTwo<Alignment>
class StorageBuffer {
public:
  explicit StorageBuffer(std::size_t size) : m_size{size}, m_ptr{nullptr} {
    if (m_size > 0) {
      m_ptr = do_allocate(m_size);

      if (reinterpret_cast<std::uintptr_t>(m_ptr) % Alignment != 0) {
        do_free(m_ptr);
        m_ptr = nullptr;

        throw std::bad_alloc();
      }
    }
  }

  ~StorageBuffer() noexcept {
    if (m_ptr) {
      do_free(m_ptr);
    }
  }

  StorageBuffer(const StorageBuffer&) = delete;
  StorageBuffer& operator=(const StorageBuffer&) = delete;

  StorageBuffer(StorageBuffer&& other) noexcept
      : m_size{std::exchange(other.m_size, 0)}, m_ptr{std::exchange(other.m_ptr, nullptr)} {}

  StorageBuffer& operator=(StorageBuffer&& other) noexcept {
    StorageBuffer temp{std::move(other)};
    swap(temp);
    return *this;
  }

  void swap(StorageBuffer& other) noexcept {
    using std::swap;
    swap(m_size, other.m_size);
    swap(m_ptr, other.m_ptr);
  }

  friend void swap(StorageBuffer& a, StorageBuffer& b) noexcept {
    a.swap(b);
  }

  [[nodiscard]] auto ptr() noexcept -> void* {
    return m_ptr;
  }

  [[nodiscard]] auto ptr() const noexcept -> const void* {
    return m_ptr;
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t {
    return m_size;
  }

private:
  static auto do_allocate(std::size_t size) -> void* {
    return Alloc::allocate(size, static_cast<std::align_val_t>(Alignment));
  }

  static auto do_free(void* ptr) noexcept -> void {
    Alloc::free(ptr, static_cast<std::align_val_t>(Alignment));
  }

  std::size_t m_size;
  void* m_ptr;
};

}  // namespace alpack