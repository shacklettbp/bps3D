#pragma once

#include <memory>

namespace bps3D {

template <typename T>
struct HandleDeleter {
    constexpr HandleDeleter() noexcept = default;
    void operator()(std::remove_extent_t<T> *ptr) const;
};

template <typename T>
using Handle = std::unique_ptr<T, HandleDeleter<T>>;

}
