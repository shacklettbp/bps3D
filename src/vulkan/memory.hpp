#pragma once

#include <glm/glm.hpp>

#include <atomic>
#include <utility>

#include "utils.hpp"
#include "core.hpp"

namespace bps3D {
namespace vk {

class MemoryAllocator;

template <bool host_mapped>
class AllocDeleter {
public:
    AllocDeleter(VkDeviceMemory mem, MemoryAllocator &alloc)
        : mem_(mem),
          alloc_(alloc)
    {}

    void operator()(VkBuffer buffer) const;
    void operator()(VkImage image) const;

    void clear();

private:
    VkDeviceMemory mem_;

    MemoryAllocator &alloc_;
};

class HostBuffer {
public:
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer(HostBuffer &&o);
    ~HostBuffer();

    void flush(const DeviceState &dev);
    void flush(const DeviceState &dev,
               VkDeviceSize offset,
               VkDeviceSize num_bytes);

    VkBuffer buffer;
    void *ptr;

private:
    HostBuffer(VkBuffer buf,
               void *p,
               VkMappedMemoryRange mem_range,
               AllocDeleter<true> deleter);

    const VkMappedMemoryRange mem_range_;

    AllocDeleter<true> deleter_;
    friend class MemoryAllocator;
};

class LocalBuffer {
public:
    LocalBuffer(const LocalBuffer &) = delete;
    LocalBuffer(LocalBuffer &&o);
    ~LocalBuffer();

    VkBuffer buffer;

private:
    LocalBuffer(VkBuffer buf, AllocDeleter<false> deleter);

    AllocDeleter<false> deleter_;
    friend class MemoryAllocator;
};

class LocalImage {
public:
    LocalImage(const LocalImage &) = delete;
    LocalImage(LocalImage &&o);
    ~LocalImage();

    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    VkImage image;

private:
    LocalImage(uint32_t width,
               uint32_t height,
               uint32_t mip_levels,
               VkImage image,
               AllocDeleter<false> deleter);

    AllocDeleter<false> deleter_;
    friend class MemoryAllocator;
};

struct LocalTexture {
    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    VkImage image;
};

struct MemoryChunk {
    VkDeviceMemory hdl;
    uint32_t offset;
    uint32_t chunkID;
};

struct MemoryTypeIndices {
    uint32_t host;
    uint32_t local;
    uint32_t dedicatedBuffer;
    uint32_t colorAttachment;
    uint32_t depthAttachment;
};

struct ResourceFormats {
    VkFormat texture;
    VkFormat colorAttachment;
    VkFormat depthAttachment;
    VkFormat linearDepthAttachment;
};

struct Alignments {
    VkDeviceSize uniformBuffer;
    VkDeviceSize storageBuffer;
};

struct TextureRequirements {
    VkDeviceSize alignment;
    VkDeviceSize size;
};

class MemoryAllocator {
public:
    MemoryAllocator(const DeviceState &dev, const InstanceState &inst);
    MemoryAllocator(const MemoryAllocator &) = delete;
    MemoryAllocator(MemoryAllocator &&) = default;

    HostBuffer makeStagingBuffer(VkDeviceSize num_bytes);
    HostBuffer makeParamBuffer(VkDeviceSize num_bytes);

    std::optional<LocalBuffer> makeIndirectBuffer(VkDeviceSize num_bytes);
    std::optional<LocalBuffer> makeLocalBuffer(VkDeviceSize num_bytes);

    std::pair<LocalBuffer, VkDeviceMemory> makeDedicatedBuffer(
        VkDeviceSize num_bytes);

    std::pair<LocalTexture, TextureRequirements>
    makeTexture(uint32_t width, uint32_t height, uint32_t mip_levels);

    void destroyTexture(LocalTexture &&texture);

    std::optional<VkDeviceMemory> alloc(VkDeviceSize num_bytes);

    LocalImage makeColorAttachment(uint32_t width, uint32_t height);
    LocalImage makeDepthAttachment(uint32_t width, uint32_t height);
    LocalImage makeLinearDepthAttachment(uint32_t width, uint32_t height);

    const ResourceFormats &getFormats() const { return formats_; }

    VkDeviceSize alignUniformBufferOffset(VkDeviceSize offset) const;
    VkDeviceSize alignStorageBufferOffset(VkDeviceSize offset) const;

private:
    HostBuffer makeHostBuffer(VkDeviceSize num_bytes,
                              VkBufferUsageFlags usage);

    std::optional<LocalBuffer> makeLocalBuffer(VkDeviceSize num_bytes,
                                               VkBufferUsageFlags usage);

    LocalImage makeDedicatedImage(uint32_t width,
                                  uint32_t height,
                                  uint32_t mip_levels,
                                  VkFormat format,
                                  VkImageUsageFlags usage,
                                  uint32_t type_idx);

    const DeviceState &dev;
    ResourceFormats formats_;
    MemoryTypeIndices type_indices_;
    Alignments alignments_;
    VkBufferUsageFlags local_buffer_usage_flags_;

    template <bool>
    friend class AllocDeleter;
};

}
}
