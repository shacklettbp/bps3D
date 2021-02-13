#pragma once

#include <bps3D/config.hpp>
#include <bps3D_core/scene.hpp>

#include <filesystem>
#include <list>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "descriptors.hpp"
#include "utils.hpp"
#include "core.hpp"
#include "memory.hpp"
#include "shader.hpp"

// Forward declare ktxTexture as kind of an opaque backing data type
struct ktxTexture;

namespace bps3D {
namespace vk {

struct VulkanScene;

struct VulkanEnvironment : public EnvironmentBackend {
    VulkanEnvironment(const Camera &cam, const VulkanScene &scene);

    uint32_t addLight(const glm::vec3 &position, const glm::vec3 &color);

    void removeLight(uint32_t light_idx);

    FrustumBounds frustumBounds;
    std::vector<PackedLight> lights;
};

struct TextureData {
    TextureData(const DeviceState &d, MemoryAllocator &a);
    TextureData(const TextureData &) = delete;
    TextureData(TextureData &&);
    ~TextureData();

    const DeviceState &dev;
    MemoryAllocator &alloc;

    VkDeviceMemory memory;
    std::vector<LocalTexture> textures;
    std::vector<VkImageView> views;
};

struct VulkanScene : public Scene {
    TextureData textures;
    DescriptorSet cullSet;
    DescriptorSet drawSet;

    LocalBuffer data;
    VkDeviceSize indexOffset;
    uint32_t numMeshes;
};

class VulkanLoader : public LoaderBackend {
public:
    VulkanLoader(const DeviceState &dev,
                 MemoryAllocator &alloc,
                 const QueueState &transfer_queue,
                 const QueueState &gfx_queue,
                 const ShaderPipeline &cull_shader,
                 const ShaderPipeline &draw_shader,
                 bool need_materials,
                 bool need_lighting);

    std::shared_ptr<Scene> loadScene(SceneLoadData &&load_info);

private:
    const DeviceState &dev;
    MemoryAllocator &alloc;
    const QueueState &transfer_queue_;
    const QueueState &gfx_queue_;

    VkCommandPool transfer_cmd_pool_;
    VkCommandBuffer transfer_stage_cmd_;
    VkCommandPool gfx_cmd_pool_;
    VkCommandBuffer gfx_copy_cmd_;

    VkSemaphore ownership_sema_;
    VkFence fence_;

    DescriptorManager cull_desc_mgr_;
    DescriptorManager draw_desc_mgr_;

    bool need_materials_;
    bool need_lighting_;
};

}
}
