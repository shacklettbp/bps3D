#pragma once

#include <glm/glm.hpp>

namespace bps3D {

enum class BackendSelect {
    Vulkan,
};

enum class RenderMode {
    Depth,
    UnlitRGB,
    ShadedRGB,
};

struct RenderConfig {
    int gpuID;
    uint32_t numLoaders;
    uint32_t batchSize;
    uint32_t imgWidth;
    uint32_t imgHeight;
    bool doubleBuffered;
    RenderMode mode;
};

inline constexpr RenderMode &operator|=(RenderMode &a, RenderMode b)
{
    return a = static_cast<RenderMode>(static_cast<uint32_t>(a) |
                                       static_cast<uint32_t>(b));
}

inline constexpr RenderMode operator|(RenderMode a, RenderMode b)
{
    return a |= b;
}

inline constexpr bool operator&(RenderMode flags, RenderMode mask)
{
    uint32_t mask_int = static_cast<uint32_t>(mask);
    return (static_cast<uint32_t>(flags) & mask_int) == mask_int;
}

}
