#pragma once
#include <core/memory/allocator.h>
#include <core/memory/storage_buffer.h>

#include <type_traits>

namespace alpack {

template <typename Object, Allocator Alloc = DefaultAllocator>
  requires std::is_trivial_v<Object>
class TrivialObjectPool {
public:
  explicit TrivialObjectPool(std::size_t size) : m_size{size}, m_storage{m_size * sizeof(Object)} {
    m_ptr = new (m_storage.ptr()) Object[m_size];
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t {
    return m_size;
  }

  [[nodiscard]] auto begin(this auto& self) noexcept {
    return self.ptr();
  }

  [[nodiscard]] auto end(this auto& self) noexcept {
    return self.ptr() + self.size();
  }

  [[nodiscard]] auto item(this auto& self, std::size_t idx) noexcept -> auto& {
    return self.ptr()[idx];
  }

  [[nodiscard]] auto span(this auto& self, std::size_t idx, std::size_t size) noexcept {
    return std::span(&self.item(idx), size);
  }

  template <std::size_t Size>
  [[nodiscard]] auto span(this auto& self, std::size_t idx) noexcept {
    return self.span(idx, Size).template first<Size>();
  }

private:
  [[nodiscard]] auto ptr(this auto& self) noexcept
    -> std::conditional_t<std::is_const_v<std::remove_reference_t<decltype(self)>>, const Object*, Object*> {
    return self.m_ptr;
  }

  std::size_t m_size;
  StorageBuffer<Alloc, alignof(Object)> m_storage;
  Object* m_ptr;
};

}  // namespace alpack