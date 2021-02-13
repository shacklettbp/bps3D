#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <bps3D/config.hpp>
#include <bps3D_core/common.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_precision.hpp>

#include "core.hpp"
#include "cuda_interop.hpp"
#include "descriptors.hpp"
#include "memory.hpp"
#include "shader.hpp"

namespace bps3D {
namespace vk {

struct BackendConfig {
    bool colorOutput;
    bool depthOutput;
    bool needMaterials;
    bool needLighting;
    uint32_t numBatches;
};

struct FramebufferConfig {
    uint32_t imgWidth;
    uint32_t imgHeight;

    uint32_t miniBatchSize;
    uint32_t numImagesWidePerMiniBatch;
    uint32_t numImagesTallPerMiniBatch;

    uint32_t numImagesWidePerBatch;
    uint32_t numImagesTallPerBatch;

    uint32_t frameWidth;
    uint32_t frameHeight;
    uint32_t totalWidth;
    uint32_t totalHeight;

    uint64_t colorLinearBytesPerBatch;
    uint64_t depthLinearBytesPerBatch;
    uint64_t linearBytesPerBatch;

    uint64_t totalLinearBytes;

    std::vector<VkClearValue> clearValues;
};

struct ParamBufferConfig {
    VkDeviceSize totalTransformBytes;

    VkDeviceSize viewOffset;
    VkDeviceSize totalViewBytes;

    VkDeviceSize materialIndicesOffset;
    VkDeviceSize totalMaterialIndexBytes;

    VkDeviceSize lightsOffset;
    VkDeviceSize totalLightParamBytes;

    VkDeviceSize cullInputOffset;
    VkDeviceSize totalCullInputBytes;

    VkDeviceSize totalParamBytes;

    VkDeviceSize countIndirectOffset;
    VkDeviceSize totalCountIndirectBytes;

    VkDeviceSize drawIndirectOffset;
    VkDeviceSize totalDrawIndirectBytes;

    VkDeviceSize totalIndirectBytes;
};

struct FramebufferState {
    std::vector<LocalImage> attachments;
    std::vector<VkImageView> attachmentViews;

    VkFramebuffer hdl;

    LocalBuffer resultBuffer;
    VkDeviceMemory resultMem;

    CudaImportedBuffer extBuffer;
};

struct RenderState {
    VkSampler textureSampler;
    VkRenderPass renderPass;

    ShaderPipeline cull;
    FixedDescriptorPool cullPool;

    ShaderPipeline draw;
    FixedDescriptorPool drawPool;
};

struct RasterPipelineState {
    VkPipelineLayout cullLayout;
    VkPipeline cullPipeline;

    VkPipelineLayout drawLayout;
    VkPipeline drawPipeline;
};

struct PipelineState {
    // Not saved (no caching)
    VkPipelineCache pipelineCache;

    RasterPipelineState rasterState;
};

struct PerBatchState {
    VkFence fence;
    std::array<VkCommandBuffer, 2> commands;
    // indirectDrawBuffer starts with batch_size draw counts,
    // followed by the actual indirect draw commands
    VkDeviceSize indirectCountBaseOffset;
    VkDeviceSize indirectCountTotalBytes;
    VkDeviceSize indirectBaseOffset;
    DynArray<uint32_t> drawOffsets;
    DynArray<uint32_t> maxNumDraws;

    glm::u32vec2 baseFBOffset;
    DynArray<glm::u32vec2> batchFBOffsets;

    VkDeviceSize colorBufferOffset;
    VkDeviceSize depthBufferOffset;

    VkDescriptorSet cullSet;
    VkDescriptorSet drawSet;

    glm::mat4x3 *transformPtr;
    ViewInfo *viewPtr;
    uint32_t *materialPtr;
    PackedLight *lightPtr;
    uint32_t *numLightsPtr;
    DrawInput *drawPtr;
};

class VulkanBackend : public RenderBackend {
public:
    VulkanBackend(const RenderConfig &cfg, bool validate);
    LoaderImpl makeLoader();

    EnvironmentImpl makeEnvironment(const Camera &cam,
                                    const std::shared_ptr<Scene> &scene);

    uint32_t render(const Environment *envs);

    void waitForFrame(uint32_t batch_idx);

    uint8_t *getColorPointer(uint32_t batch_idx);
    float *getDepthPointer(uint32_t batch_idx);

private:
    VulkanBackend(const RenderConfig &cfg,
                  const BackendConfig &backend_cfg,
                  bool validate);

    const uint32_t batch_size_;

    const InstanceState inst;
    const DeviceState dev;

    MemoryAllocator alloc;

    const FramebufferConfig fb_cfg_;
    const ParamBufferConfig param_cfg_;
    RenderState render_state_;
    PipelineState pipeline_;
    FramebufferState fb_;

    DynArray<QueueState> transfer_queues_;
    DynArray<QueueState> graphics_queues_;
    DynArray<QueueState> compute_queues_;

    HostBuffer render_input_buffer_;
    LocalBuffer indirect_draw_buffer_;

    VkCommandPool gfx_cmd_pool_;
    std::atomic_int num_loaders_;
    int max_loaders_;
    bool need_materials_;
    bool need_lighting_;
    const uint32_t mini_batch_size_;
    const uint32_t num_mini_batches_;
    glm::u32vec2 per_elem_render_size_;
    glm::u32vec2 per_minibatch_render_size_;

    std::vector<PerBatchState> batch_states_;

    int cur_batch_;
    const int batch_mask_;
};

}
}
