#include "render.hpp"

#include "scene.hpp"

#include <iostream>

using namespace std;

namespace bps3D {
namespace vk {

static BackendConfig getBackendConfig(const RenderConfig &cfg)
{
    bool need_lighting = cfg.mode & RenderMode::ShadedRGB;

    bool color_output = (cfg.mode & RenderMode::UnlitRGB) || need_lighting;
    bool depth_output = cfg.mode & RenderMode::Depth;

    bool need_materials = color_output;

    return BackendConfig {
        color_output,
        depth_output,
        need_materials,
        need_lighting,
        cfg.doubleBuffered ? 2u : 1u,
    };
}

static ParamBufferConfig getParamBufferConfig(const BackendConfig &backend_cfg,
                                              uint32_t batch_size,
                                              const MemoryAllocator &alloc)
{
    ParamBufferConfig cfg {};

    cfg.totalTransformBytes =
        sizeof(glm::mat4x3) * VulkanConfig::max_instances;

    VkDeviceSize cur_offset = cfg.totalTransformBytes;

    if (backend_cfg.needMaterials) {
        cfg.materialIndicesOffset = cur_offset;

        cfg.totalMaterialIndexBytes =
            sizeof(uint32_t) * VulkanConfig::max_instances;

        cur_offset = cfg.materialIndicesOffset + cfg.totalMaterialIndexBytes;
    }

    cfg.viewOffset = alloc.alignUniformBufferOffset(cur_offset);
    cfg.totalViewBytes = sizeof(ViewInfo) * batch_size;

    cur_offset = cfg.viewOffset + cfg.totalViewBytes;

    if (backend_cfg.needLighting) {
        cfg.lightsOffset = alloc.alignUniformBufferOffset(cur_offset);
        cfg.totalLightParamBytes =
            sizeof(PackedLight) * VulkanConfig::max_lights + sizeof(uint32_t);

        cur_offset = cfg.lightsOffset + cfg.totalLightParamBytes;
    }

    cfg.cullInputOffset = alloc.alignStorageBufferOffset(cur_offset);
    cfg.totalCullInputBytes = sizeof(DrawInput) * VulkanConfig::max_instances;
    cur_offset = cfg.cullInputOffset + cfg.totalCullInputBytes;

    // Ensure that full block is aligned to maximum requirement
    cfg.totalParamBytes = alloc.alignStorageBufferOffset(
        alloc.alignUniformBufferOffset(cur_offset));

    cfg.countIndirectOffset = 0;
    cfg.totalCountIndirectBytes = sizeof(uint32_t) * batch_size;

    cfg.drawIndirectOffset = alloc.alignStorageBufferOffset(
        alloc.alignUniformBufferOffset(cfg.totalCountIndirectBytes));
    cfg.totalDrawIndirectBytes =
        sizeof(VkDrawIndexedIndirectCommand) * VulkanConfig::max_instances;

    cfg.totalIndirectBytes =
        alloc.alignStorageBufferOffset(alloc.alignUniformBufferOffset(
            cfg.drawIndirectOffset + cfg.totalDrawIndirectBytes));

    return cfg;
}

static FramebufferConfig getFramebufferConfig(const RenderConfig &cfg,
                                              const BackendConfig &backend_cfg)
{
    uint32_t batch_size = cfg.batchSize;
    uint32_t num_batches = backend_cfg.numBatches;

    uint32_t minibatch_size =
        max(batch_size / VulkanConfig::minibatch_divisor, batch_size);
    assert(batch_size % minibatch_size == 0);

    uint32_t batch_fb_images_wide = ceil(sqrt(batch_size));
    while (batch_size % batch_fb_images_wide != 0) {
        batch_fb_images_wide++;
    }

    uint32_t minibatch_fb_images_wide;
    uint32_t minibatch_fb_images_tall;
    if (batch_fb_images_wide >= minibatch_size) {
        assert(batch_fb_images_wide % minibatch_size == 0);
        minibatch_fb_images_wide = minibatch_size;
        minibatch_fb_images_tall = 1;
    } else {
        minibatch_fb_images_wide = batch_fb_images_wide;
        minibatch_fb_images_tall = minibatch_size / batch_fb_images_wide;
    }

    assert(minibatch_fb_images_wide * minibatch_fb_images_tall ==
           minibatch_size);

    uint32_t batch_fb_images_tall = (batch_size / batch_fb_images_wide);
    assert(batch_fb_images_wide * batch_fb_images_tall == batch_size);

    uint32_t batch_fb_width = cfg.imgWidth * batch_fb_images_wide;
    uint32_t batch_fb_height = cfg.imgHeight * batch_fb_images_tall;

    uint32_t total_fb_width = batch_fb_width * num_batches;
    uint32_t total_fb_height = batch_fb_height;

    vector<VkClearValue> clear_vals;

    uint64_t frame_color_bytes = 0;
    if (backend_cfg.colorOutput) {
        frame_color_bytes =
            4 * sizeof(uint8_t) * batch_fb_width * batch_fb_height;

        VkClearValue clear_val;
        clear_val.color = {{0.f, 0.f, 0.f, 1.f}};

        clear_vals.push_back(clear_val);
    }

    uint64_t frame_depth_bytes = 0;
    if (backend_cfg.depthOutput) {
        frame_depth_bytes = sizeof(float) * batch_fb_width * batch_fb_height;

        VkClearValue clear_val;
        clear_val.color = {{0.f, 0.f, 0.f, 0.f}};

        clear_vals.push_back(clear_val);
    }

    VkClearValue depth_clear_value;
    depth_clear_value.depthStencil = {1.f, 0};

    clear_vals.push_back(depth_clear_value);

    uint32_t frame_linear_bytes = frame_color_bytes + frame_depth_bytes;

    assert(frame_linear_bytes > 0);

    return FramebufferConfig {cfg.imgWidth,
                              cfg.imgHeight,
                              minibatch_size,
                              minibatch_fb_images_wide,
                              minibatch_fb_images_tall,
                              batch_fb_images_wide,
                              batch_fb_images_tall,
                              batch_fb_width,
                              batch_fb_height,
                              total_fb_width,
                              total_fb_height,
                              frame_color_bytes,
                              frame_depth_bytes,
                              frame_linear_bytes,
                              frame_linear_bytes * num_batches,
                              move(clear_vals)};
}

static VkRenderPass makeRenderPass(const DeviceState &dev,
                                   const ResourceFormats &fmts,
                                   bool color_output,
                                   bool depth_output)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    if (color_output) {
        attachment_descs.push_back(
            {0, fmts.colorAttachment, VK_SAMPLE_COUNT_1_BIT,
             VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});

        attachment_refs.push_back(
            {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    }

    if (depth_output) {
        attachment_descs.push_back(
            {0, fmts.linearDepthAttachment, VK_SAMPLE_COUNT_1_BIT,
             VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});

        attachment_refs.push_back(
            {static_cast<uint32_t>(attachment_refs.size()),
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    }

    attachment_descs.push_back(
        {0, fmts.depthAttachment, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {static_cast<uint32_t>(attachment_refs.size()),
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount =
        static_cast<uint32_t>(attachment_refs.size() - 1);
    subpass_desc.pColorAttachments = &attachment_refs[0];
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachment_descs.size());
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;
    render_pass_info.dependencyCount = 0;
    render_pass_info.pDependencies = nullptr;

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info, nullptr,
                                   &render_pass));

    return render_pass;
}

static VkSampler makeImmutableSampler(const DeviceState &dev)
{
    VkSampler sampler;

    VkSamplerCreateInfo sampler_info;
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.pNext = nullptr;
    sampler_info.flags = 0;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.mipLodBias = 0;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 0;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.minLod = 0;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    REQ_VK(dev.dt.createSampler(dev.hdl, &sampler_info, nullptr, &sampler));

    return sampler;
}

static RenderState makeRenderState(const DeviceState &dev,
                                   const BackendConfig &backend_cfg,
                                   MemoryAllocator &alloc)
{
    VkSampler texture_sampler = VK_NULL_HANDLE;
    if (backend_cfg.needMaterials) {
        texture_sampler = makeImmutableSampler(dev);
    }

    vector<string> shader_defines;

    uint32_t cur_attachment = 0;
    if (backend_cfg.colorOutput) {
        shader_defines.push_back("OUTPUT_COLOR");
        shader_defines.push_back("COLOR_ATTACHMENT " +
                                 to_string(cur_attachment++));
    }

    if (backend_cfg.depthOutput) {
        shader_defines.push_back("OUTPUT_DEPTH");

        shader_defines.push_back("DEPTH_ATTACHMENT " +
                                 to_string(cur_attachment++));
    }

    if (backend_cfg.needLighting) {
        shader_defines.push_back("LIGHTING");
    }

    if (backend_cfg.needMaterials) {
        shader_defines.push_back("MATERIALS");
    }

    ShaderPipeline::initCompiler();

    ShaderPipeline cull_shader(dev, {"meshcull.comp"}, {}, shader_defines);

    FixedDescriptorPool cull_pool(dev, cull_shader, 0, backend_cfg.numBatches);

    ShaderPipeline draw_shader(
        dev, {"uber.vert", "uber.frag"},
        {
            {1, 1, texture_sampler, 1, 0},
            {1, 2, VK_NULL_HANDLE, VulkanConfig::max_materials,
             VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},
        },
        shader_defines);

    FixedDescriptorPool draw_pool(dev, draw_shader, 0, backend_cfg.numBatches);

    return RenderState {
        texture_sampler,
        makeRenderPass(dev, alloc.getFormats(), backend_cfg.colorOutput,
                       backend_cfg.depthOutput),
        move(cull_shader),
        move(cull_pool),
        move(draw_shader),
        move(draw_pool),
    };
}

static PipelineState makePipeline(const DeviceState &dev,
                                  const BackendConfig &backend_cfg,
                                  const FramebufferConfig &fb_cfg,
                                  const RenderState &render_state)
{
    // Pipeline cache (unsaved)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info, nullptr,
                                      &pipeline_cache));

    VkPipelineVertexInputStateCreateInfo vert_info;
    vert_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount = 0;
    vert_info.pVertexBindingDescriptions = nullptr;
    vert_info.vertexAttributeDescriptionCount = 0;
    vert_info.pVertexAttributeDescriptions = nullptr;

    // Assembly
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    input_assembly_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_info.primitiveRestartEnable = VK_FALSE;

    // Viewport
    VkRect2D scissors {{0, 0}, {fb_cfg.totalWidth, fb_cfg.totalHeight}};

    VkPipelineViewportStateCreateInfo viewport_info {};
    viewport_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = nullptr;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = &scissors;

    // Multisample
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    multisample_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_info.sampleShadingEnable = VK_FALSE;
    multisample_info.alphaToCoverageEnable = VK_FALSE;
    multisample_info.alphaToOneEnable = VK_FALSE;

    // Rasterization
    VkPipelineRasterizationStateCreateInfo raster_info {};
    raster_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.depthClampEnable = VK_FALSE;
    raster_info.rasterizerDiscardEnable = VK_FALSE;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster_info.depthBiasEnable = VK_FALSE;
    raster_info.lineWidth = 1.0f;

    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info {};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depth_info.depthBoundsTestEnable = VK_FALSE;
    depth_info.stencilTestEnable = VK_FALSE;
    depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach {};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    vector<VkPipelineColorBlendAttachmentState> blend_attachments;
    if (backend_cfg.colorOutput) {
        blend_attachments.push_back(blend_attach);
    }

    if (backend_cfg.depthOutput) {
        blend_attachments.push_back(blend_attach);
    }

    VkPipelineColorBlendStateCreateInfo blend_info {};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount =
        static_cast<uint32_t>(blend_attachments.size());
    blend_info.pAttachments = blend_attachments.data();

    // Dynamic
    VkDynamicState dyn_viewport_enable = VK_DYNAMIC_STATE_VIEWPORT;

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = 1;
    dyn_info.pDynamicStates = &dyn_viewport_enable;

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(DrawPushConstant),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 2> draw_desc_layouts {{
        render_state.draw.getLayout(0),
        render_state.draw.getLayout(1),
    }};

    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(draw_desc_layouts.size());
    gfx_layout_info.pSetLayouts = draw_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 1;
    gfx_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout draw_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &gfx_layout_info, nullptr,
                                       &draw_layout));

    array<VkPipelineShaderStageCreateInfo, 2> gfx_stages {{
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            render_state.draw.getShader(0),
            "main",
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            render_state.draw.getShader(1),
            "main",
            nullptr,
        },
    }};

    VkGraphicsPipelineCreateInfo gfx_info;
    gfx_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_info.pNext = nullptr;
    gfx_info.flags = 0;
    gfx_info.stageCount = gfx_stages.size();
    gfx_info.pStages = gfx_stages.data();
    gfx_info.pVertexInputState = &vert_info;
    gfx_info.pInputAssemblyState = &input_assembly_info;
    gfx_info.pTessellationState = nullptr;
    gfx_info.pViewportState = &viewport_info;
    gfx_info.pRasterizationState = &raster_info;
    gfx_info.pMultisampleState = &multisample_info;
    gfx_info.pDepthStencilState = &depth_info;
    gfx_info.pColorBlendState = &blend_info;
    gfx_info.pDynamicState = &dyn_info;
    gfx_info.layout = draw_layout;
    gfx_info.renderPass = render_state.renderPass;
    gfx_info.subpass = 0;
    gfx_info.basePipelineHandle = VK_NULL_HANDLE;
    gfx_info.basePipelineIndex = -1;

    VkPipeline draw_pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &gfx_info, nullptr, &draw_pipeline));

    // Compute shaders for culling
    array<VkDescriptorSetLayout, 2> cull_desc_layouts {
        render_state.cull.getLayout(0),
        render_state.cull.getLayout(1),
    };

    VkPushConstantRange cull_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(CullPushConstant),
    };

    VkPipelineLayoutCreateInfo cull_layout_info;
    cull_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    cull_layout_info.pNext = nullptr;
    cull_layout_info.flags = 0;
    cull_layout_info.setLayoutCount = cull_desc_layouts.size();
    cull_layout_info.pSetLayouts = cull_desc_layouts.data();
    cull_layout_info.pushConstantRangeCount = 1;
    cull_layout_info.pPushConstantRanges = &cull_const;

    VkPipelineLayout cull_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &cull_layout_info, nullptr,
                                       &cull_layout));

    VkComputePipelineCreateInfo cull_compute_info;
    cull_compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cull_compute_info.pNext = nullptr;
    cull_compute_info.flags = 0;
    cull_compute_info.stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        render_state.cull.getShader(0),
        "main",
        nullptr,
    };
    cull_compute_info.layout = cull_layout;
    cull_compute_info.basePipelineHandle = VK_NULL_HANDLE;
    cull_compute_info.basePipelineIndex = -1;

    VkPipeline cull_pipeline;
    REQ_VK(dev.dt.createComputePipelines(dev.hdl, pipeline_cache, 1,
                                         &cull_compute_info, nullptr,
                                         &cull_pipeline));

    return PipelineState {
        pipeline_cache,
        RasterPipelineState {
            cull_layout,
            cull_pipeline,
            draw_layout,
            draw_pipeline,
        },
    };
}

static FramebufferState makeFramebuffer(const DeviceState &dev,
                                        const RenderConfig &cfg,
                                        const BackendConfig &backend_cfg,
                                        const FramebufferConfig &fb_cfg,
                                        MemoryAllocator &alloc,
                                        VkRenderPass render_pass)
{
    vector<LocalImage> attachments;
    vector<VkImageView> attachment_views;

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    if (backend_cfg.colorOutput) {
        attachments.emplace_back(
            alloc.makeColorAttachment(fb_cfg.totalWidth, fb_cfg.totalHeight));

        VkImageView color_view;
        view_info.image = attachments.back().image;
        view_info.format = alloc.getFormats().colorAttachment;

        REQ_VK(
            dev.dt.createImageView(dev.hdl, &view_info, nullptr, &color_view));

        attachment_views.push_back(color_view);
    }

    if (backend_cfg.depthOutput) {
        attachments.emplace_back(alloc.makeLinearDepthAttachment(
            fb_cfg.totalWidth, fb_cfg.totalHeight));

        VkImageView linear_depth_view;
        view_info.image = attachments.back().image;
        view_info.format = alloc.getFormats().linearDepthAttachment;

        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr,
                                      &linear_depth_view));

        attachment_views.push_back(linear_depth_view);
    }

    VkFramebuffer fb_handle = VK_NULL_HANDLE;
    attachments.emplace_back(
        alloc.makeDepthAttachment(fb_cfg.totalWidth, fb_cfg.totalHeight));

    view_info.image = attachments.back().image;
    view_info.format = alloc.getFormats().depthAttachment;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &depth_view));
    attachment_views.push_back(depth_view);

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(attachment_views.size());
    fb_info.pAttachments = attachment_views.data();
    fb_info.width = fb_cfg.totalWidth;
    fb_info.height = fb_cfg.totalHeight;
    fb_info.layers = 1;

    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &fb_handle));

    auto [result_buffer, result_mem] =
        alloc.makeDedicatedBuffer(fb_cfg.totalLinearBytes);

    return FramebufferState {
        move(attachments),
        attachment_views,
        fb_handle,
        move(result_buffer),
        result_mem,
        CudaImportedBuffer(dev, cfg.gpuID, result_mem,
                           fb_cfg.totalLinearBytes),
    };
}

static void recordFBToLinearCopy(const DeviceState &dev,
                                 const BackendConfig &backend_cfg,
                                 const PerBatchState &state,
                                 const FramebufferConfig &fb_cfg,
                                 const FramebufferState &fb)
{
    vector<VkImageMemoryBarrier> fb_barriers;

    uint32_t view_offset = 0;
    if (backend_cfg.colorOutput) {
        fb_barriers.emplace_back(
            VkImageMemoryBarrier {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                  nullptr,
                                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_ACCESS_TRANSFER_READ_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_QUEUE_FAMILY_IGNORED,
                                  VK_QUEUE_FAMILY_IGNORED,
                                  fb.attachments[view_offset].image,
                                  {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});

        view_offset++;
    }

    if (backend_cfg.depthOutput) {
        fb_barriers.emplace_back(
            VkImageMemoryBarrier {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                  nullptr,
                                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                  VK_ACCESS_TRANSFER_READ_BIT,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_QUEUE_FAMILY_IGNORED,
                                  VK_QUEUE_FAMILY_IGNORED,
                                  fb.attachments[view_offset].image,
                                  {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});

        view_offset++;
    }

    VkCommandBuffer copy_cmd = state.commands[1];

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(copy_cmd, &begin_info));
    dev.dt.cmdPipelineBarrier(
        copy_cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0,
        nullptr, 0, nullptr, static_cast<uint32_t>(fb_barriers.size()),
        fb_barriers.data());

    uint32_t batch_size = state.batchFBOffsets.size();

    DynArray<VkBufferImageCopy> copy_regions(batch_size);

    auto make_copy_cmd = [&](VkDeviceSize base_offset, uint32_t texel_bytes,
                             VkImage src_image) {
        uint32_t cur_offset = base_offset;

        for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            glm::u32vec2 cur_fb_pos = state.batchFBOffsets[batch_idx];

            VkBufferImageCopy &region = copy_regions[batch_idx];
            region.bufferOffset = cur_offset;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageOffset = {static_cast<int32_t>(cur_fb_pos.x),
                                  static_cast<int32_t>(cur_fb_pos.y), 0};
            region.imageExtent = {fb_cfg.imgWidth, fb_cfg.imgHeight, 1};

            cur_offset += fb_cfg.imgWidth * fb_cfg.imgHeight * texel_bytes;
        }

        dev.dt.cmdCopyImageToBuffer(
            copy_cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            fb.resultBuffer.buffer, batch_size, copy_regions.data());
    };

    uint32_t attachment_offset = 0;
    if (backend_cfg.colorOutput) {
        make_copy_cmd(state.colorBufferOffset, sizeof(uint8_t) * 4,
                      fb.attachments[attachment_offset].image);

        attachment_offset++;
    }

    if (backend_cfg.depthOutput) {
        make_copy_cmd(state.depthBufferOffset, sizeof(float),
                      fb.attachments[attachment_offset].image);

        attachment_offset++;
    }

    REQ_VK(dev.dt.endCommandBuffer(copy_cmd));
}

static PerBatchState makePerBatchState(const DeviceState &dev,
                                       const BackendConfig &backend_cfg,
                                       const FramebufferConfig &fb_cfg,
                                       const ParamBufferConfig &param_cfg,
                                       VkCommandPool gfx_cmd_pool,
                                       HostBuffer &param_buffer,
                                       LocalBuffer &indirect_buffer,
                                       VkDescriptorSet cull_set,
                                       VkDescriptorSet draw_set,
                                       uint32_t batch_size,
                                       uint32_t global_batch_idx)
{
    auto computeFBPosition = [&fb_cfg](uint32_t batch_idx) {
        return glm::u32vec2(
            (batch_idx % fb_cfg.numImagesWidePerBatch) * fb_cfg.imgWidth,
            (batch_idx / fb_cfg.numImagesWidePerBatch) * fb_cfg.imgHeight);
    };

    VkCommandBuffer draw_command = makeCmdBuffer(dev, gfx_cmd_pool);
    VkCommandBuffer copy_command = makeCmdBuffer(dev, gfx_cmd_pool);

    glm::u32vec2 base_fb_offset(
        global_batch_idx * fb_cfg.numImagesWidePerBatch * fb_cfg.imgWidth, 0);

    DynArray<glm::u32vec2> batch_fb_offsets(batch_size);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        batch_fb_offsets[batch_idx] =
            computeFBPosition(batch_idx) + base_fb_offset;
    }

    VkDeviceSize color_buffer_offset =
        global_batch_idx * fb_cfg.linearBytesPerBatch;

    VkDeviceSize depth_buffer_offset =
        color_buffer_offset + fb_cfg.colorLinearBytesPerBatch;

    vector<VkWriteDescriptorSet> desc_set_updates;

    VkDeviceSize base_offset = global_batch_idx * param_cfg.totalParamBytes;

    uint8_t *base_ptr =
        reinterpret_cast<uint8_t *>(param_buffer.ptr) + base_offset;

    glm::mat4x3 *transform_ptr = reinterpret_cast<glm::mat4x3 *>(base_ptr);

    ViewInfo *view_ptr =
        reinterpret_cast<ViewInfo *>(base_ptr + param_cfg.viewOffset);

    uint32_t *material_ptr = nullptr;
    PackedLight *light_ptr = nullptr;
    uint32_t *num_lights_ptr = nullptr;

    if (backend_cfg.needMaterials) {
        material_ptr = reinterpret_cast<uint32_t *>(
            base_ptr + param_cfg.materialIndicesOffset);
    }

    if (backend_cfg.needLighting) {
        light_ptr =
            reinterpret_cast<PackedLight *>(base_ptr + param_cfg.lightsOffset);

        num_lights_ptr =
            reinterpret_cast<uint32_t *>(light_ptr + VulkanConfig::max_lights);
    }

    VkDeviceSize base_indirect_offset =
        param_cfg.totalIndirectBytes * global_batch_idx;
    VkDeviceSize count_indirect_offset =
        base_indirect_offset + param_cfg.countIndirectOffset;
    VkDeviceSize draw_indirect_offset =
        base_indirect_offset + param_cfg.drawIndirectOffset;

    DrawInput *draw_ptr =
        reinterpret_cast<DrawInput *>(base_ptr + param_cfg.cullInputOffset);

    DescriptorUpdates desc_updates(8);

    // Cull set

    VkDescriptorBufferInfo transform_info {
        param_buffer.buffer,
        base_offset,
        param_cfg.totalTransformBytes,
    };

    desc_updates.buffer(cull_set, &transform_info, 0,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo view_buffer_info {
        param_buffer.buffer,
        base_offset + param_cfg.viewOffset,
        param_cfg.totalViewBytes,
    };

    desc_updates.buffer(cull_set, &view_buffer_info, 1,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo indirect_input_buffer_info {
        param_buffer.buffer,
        param_cfg.cullInputOffset,
        param_cfg.totalCullInputBytes,
    };

    desc_updates.buffer(cull_set, &indirect_input_buffer_info, 2,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo indirect_output_buffer_info {
        indirect_buffer.buffer, draw_indirect_offset,
        param_cfg.totalDrawIndirectBytes};

    desc_updates.buffer(cull_set, &indirect_output_buffer_info, 3,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo indirect_count_buffer_info {
        indirect_buffer.buffer, count_indirect_offset,
        param_cfg.totalCountIndirectBytes};

    desc_updates.buffer(cull_set, &indirect_count_buffer_info, 4,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    // Draw set

    desc_updates.buffer(draw_set, &view_buffer_info, 0,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    desc_updates.buffer(draw_set, &transform_info, 1,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorBufferInfo mat_info;
    if (material_ptr) {
        mat_info = {
            param_buffer.buffer,
            base_offset + param_cfg.materialIndicesOffset,
            param_cfg.totalMaterialIndexBytes,
        };
        desc_updates.buffer(draw_set, &mat_info, 2,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }

    VkDescriptorBufferInfo light_info;
    if (light_ptr) {
        light_info = {
            param_buffer.buffer,
            base_offset + param_cfg.lightsOffset,
            param_cfg.totalLightParamBytes,
        };

        desc_updates.buffer(draw_set, &light_info, 3,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    }

    desc_updates.update(dev);

    return PerBatchState {makeFence(dev),
                          {draw_command, copy_command},
                          count_indirect_offset,
                          sizeof(uint32_t) * batch_size,
                          draw_indirect_offset,
                          DynArray<uint32_t>(batch_size),
                          DynArray<uint32_t>(batch_size),
                          base_fb_offset,
                          move(batch_fb_offsets),
                          color_buffer_offset,
                          depth_buffer_offset,
                          cull_set,
                          draw_set,
                          transform_ptr,
                          view_ptr,
                          material_ptr,
                          light_ptr,
                          num_lights_ptr,
                          draw_ptr};
}

VulkanBackend::VulkanBackend(const RenderConfig &cfg, bool validate)
    : VulkanBackend(cfg, getBackendConfig(cfg), validate)
{}

VulkanBackend::VulkanBackend(const RenderConfig &cfg,
                             const BackendConfig &backend_cfg,
                             bool validate)
    : batch_size_(cfg.batchSize),
      inst(validate, false, {}),
      dev(inst.makeDevice(getUUIDFromCudaID(cfg.gpuID),
                          false,
                          2,
                          1,
                          cfg.numLoaders,
                          nullptr)),
      alloc(dev, inst),
      fb_cfg_(getFramebufferConfig(cfg, backend_cfg)),
      param_cfg_(getParamBufferConfig(backend_cfg, cfg.batchSize, alloc)),
      render_state_(makeRenderState(dev, backend_cfg, alloc)),
      pipeline_(makePipeline(dev, backend_cfg, fb_cfg_, render_state_)),
      fb_(makeFramebuffer(dev,
                          cfg,
                          backend_cfg,
                          fb_cfg_,
                          alloc,
                          render_state_.renderPass)),
      transfer_queues_(dev.numTransferQueues),
      graphics_queues_(dev.numGraphicsQueues),
      compute_queues_(dev.numComputeQueues),
      render_input_buffer_(alloc.makeParamBuffer(param_cfg_.totalParamBytes *
                                                 backend_cfg.numBatches)),
      indirect_draw_buffer_([&]() {
          auto opt_buffer = alloc.makeIndirectBuffer(
              param_cfg_.totalIndirectBytes * backend_cfg.numBatches);
          if (!opt_buffer.has_value()) {
              cerr << "Vulkan: Out of device memory during initialization"
                   << endl;
          }

          return move(opt_buffer.value());
      }()),
      gfx_cmd_pool_(makeCmdPool(dev, dev.gfxQF)),
      num_loaders_(0),
      max_loaders_(cfg.numLoaders),
      need_materials_(backend_cfg.needMaterials),
      need_lighting_(backend_cfg.needLighting),
      mini_batch_size_(fb_cfg_.miniBatchSize),
      num_mini_batches_(batch_size_ / mini_batch_size_),
      per_elem_render_size_(fb_cfg_.imgWidth, fb_cfg_.imgHeight),
      per_minibatch_render_size_(
          per_elem_render_size_.x * fb_cfg_.numImagesWidePerMiniBatch,
          per_elem_render_size_.y * fb_cfg_.numImagesTallPerMiniBatch),
      batch_states_(),
      cur_batch_(0),
      batch_mask_(backend_cfg.numBatches == 2 ? 1 : 0)
{
    bool transfer_shared = cfg.numLoaders > dev.numTransferQueues;

    for (int i = 0; i < (int)transfer_queues_.size(); i++) {
        new (&transfer_queues_[i])
            QueueState(makeQueue(dev, dev.transferQF, i), transfer_shared);
    }

    for (int i = 0; i < (int)graphics_queues_.size(); i++) {
        new (&graphics_queues_[i])
            QueueState(makeQueue(dev, dev.gfxQF, i),
                       i == (int)graphics_queues_.size() - 1 ? true : false);
    }

    for (int i = 0; i < (int)compute_queues_.size(); i++) {
        new (&compute_queues_[i])
            QueueState(makeQueue(dev, dev.computeQF, i), false);
    }

    batch_states_.reserve(backend_cfg.numBatches);
    for (int i = 0; i < (int)backend_cfg.numBatches; i++) {
        batch_states_.emplace_back(makePerBatchState(
            dev, backend_cfg, fb_cfg_, param_cfg_, gfx_cmd_pool_,
            render_input_buffer_, indirect_draw_buffer_,
            render_state_.cullPool.makeSet(), render_state_.drawPool.makeSet(),
            cfg.batchSize, i));

        recordFBToLinearCopy(dev, backend_cfg, batch_states_.back(), fb_cfg_,
                             fb_);
    }
}

LoaderImpl VulkanBackend::makeLoader()
{
    int loader_idx = num_loaders_.fetch_add(1, memory_order_acq_rel);
    assert(loader_idx < max_loaders_);

    auto loader = new VulkanLoader(
        dev, alloc, transfer_queues_[loader_idx % transfer_queues_.size()],
        graphics_queues_.back(), render_state_.cull, render_state_.draw,
        need_materials_, need_lighting_);

    return makeLoaderImpl<VulkanLoader>(loader);
}

EnvironmentImpl VulkanBackend::makeEnvironment(const Camera &cam,
                                               const shared_ptr<Scene> &scene)
{
    const VulkanScene &vk_scene = *static_cast<VulkanScene *>(scene.get());
    VulkanEnvironment *environment = new VulkanEnvironment(cam, vk_scene);
    return makeEnvironmentImpl<VulkanEnvironment>(environment);
}

uint32_t VulkanBackend::render(const Environment *envs)
{
    PerBatchState &batch_state = batch_states_[cur_batch_];

    VkCommandBuffer render_cmd = batch_state.commands[0];

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(render_cmd, &begin_info));

    dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_.rasterState.cullPipeline);

    dev.dt.cmdBindDescriptorSets(render_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline_.rasterState.cullLayout, 0, 1,
                                 &batch_state.cullSet, 0, nullptr);

    dev.dt.cmdBindPipeline(render_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_.rasterState.drawPipeline);

    dev.dt.cmdBindDescriptorSets(render_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_.rasterState.drawLayout, 0, 1,
                                 &batch_state.drawSet, 0, nullptr);

    // Reset count buffer
    dev.dt.cmdFillBuffer(render_cmd, indirect_draw_buffer_.buffer,
                         batch_state.indirectCountBaseOffset,
                         batch_state.indirectCountTotalBytes, 0);

    VkBufferMemoryBarrier init_barrier;
    init_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    init_barrier.pNext = nullptr;
    init_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    init_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    init_barrier.srcQueueFamilyIndex = 0;
    init_barrier.dstQueueFamilyIndex = 0;
    init_barrier.buffer = indirect_draw_buffer_.buffer;
    init_barrier.offset = 0;
    init_barrier.size = VK_WHOLE_SIZE;

    dev.dt.cmdPipelineBarrier(render_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                              nullptr, 1, &init_barrier, 0, nullptr);

    // CPU-side input setup

    uint32_t draw_id = 0;
    uint32_t inst_offset = 0;
    glm::mat4x3 *transform_ptr = batch_state.transformPtr;
    uint32_t *material_ptr = batch_state.materialPtr;
    PackedLight *light_ptr = batch_state.lightPtr;
    ViewInfo *view_ptr = batch_state.viewPtr;
    for (int batch_idx = 0; batch_idx < (int)batch_size_; batch_idx++) {
        const Environment &env = envs[batch_idx];
        const VulkanEnvironment &env_backend =
            *static_cast<const VulkanEnvironment *>(env.getBackend());
        const VulkanScene &scene =
            *static_cast<const VulkanScene *>(env.getScene().get());
        const auto &env_transforms = env.getTransforms();
        const auto &env_materials = env.getMaterials();

        view_ptr->view = env.getCamera().worldToCamera;
        view_ptr->projection = env.getCamera().proj;
        view_ptr++;

        batch_state.drawOffsets[batch_idx] = draw_id;

        for (int mesh_idx = 0; mesh_idx < (int)scene.numMeshes; mesh_idx++) {
            const MeshInfo &mesh_metadata = scene.meshInfo[mesh_idx];
            uint32_t num_instances = env_transforms[mesh_idx].size();

            for (uint32_t inst_idx = 0; inst_idx < num_instances; inst_idx++) {
                for (uint32_t chunk_id = 0; chunk_id < mesh_metadata.numChunks;
                     chunk_id++) {
                    batch_state.drawPtr[draw_id] = DrawInput {
                        inst_idx + inst_offset,
                        chunk_id + mesh_metadata.chunkOffset,
                    };
                    draw_id++;
                }
            }
            inst_offset += num_instances;

            memcpy(transform_ptr, env_transforms[mesh_idx].data(),
                   sizeof(glm::mat4x3) * num_instances);

            transform_ptr += num_instances;

            if (material_ptr) {
                memcpy(material_ptr, env_materials[mesh_idx].data(),
                       num_instances * sizeof(uint32_t));

                material_ptr += num_instances;
            }
        }

        if (light_ptr) {
            uint32_t num_lights = env_backend.lights.size();

            memcpy(light_ptr, env_backend.lights.data(),
                   num_lights * sizeof(PackedLight));

            *batch_state.numLightsPtr = num_lights;

            light_ptr += num_lights;
        }

        batch_state.maxNumDraws[batch_idx] =
            draw_id - batch_state.drawOffsets[batch_idx];
    }

    uint32_t total_draws = draw_id;

    assert(total_draws < VulkanConfig::max_instances);

    VkRenderPassBeginInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.renderPass = render_state_.renderPass;
    render_pass_info.framebuffer = fb_.hdl;
    render_pass_info.clearValueCount =
        static_cast<uint32_t>(fb_cfg_.clearValues.size());
    render_pass_info.pClearValues = fb_cfg_.clearValues.data();

    // 1 indirect draw per batch elem
    uint32_t global_batch_offset = 0;
    for (int mini_batch_idx = 0; mini_batch_idx < (int)num_mini_batches_;
         mini_batch_idx++) {
        // Record culling for this mini batch
        for (int local_batch_idx = 0; local_batch_idx < (int)mini_batch_size_;
             local_batch_idx++) {
            uint32_t batch_idx = global_batch_offset + local_batch_idx;
            const Environment &env = envs[batch_idx];
            const VulkanEnvironment &env_backend =
                *static_cast<const VulkanEnvironment *>(env.getBackend());
            const VulkanScene &scene =
                *static_cast<const VulkanScene *>(env.getScene().get());

            dev.dt.cmdBindDescriptorSets(render_cmd,
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         pipeline_.rasterState.cullLayout, 1,
                                         1, &scene.cullSet.hdl, 0, nullptr);

            CullPushConstant cull_const {env_backend.frustumBounds, batch_idx,
                                         batch_state.drawOffsets[batch_idx],
                                         batch_state.maxNumDraws[batch_idx]};

            dev.dt.cmdPushConstants(render_cmd,
                                    pipeline_.rasterState.cullLayout,
                                    VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    sizeof(CullPushConstant), &cull_const);

            dev.dt.cmdDispatch(
                render_cmd,
                getWorkgroupSize(batch_state.maxNumDraws[batch_idx]), 1, 1);
        }

        // Cull / render barrier
        VkBufferMemoryBarrier buffer_barrier;
        buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buffer_barrier.pNext = nullptr;
        buffer_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        buffer_barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        buffer_barrier.srcQueueFamilyIndex = 0;
        buffer_barrier.dstQueueFamilyIndex = 0;
        buffer_barrier.buffer = indirect_draw_buffer_.buffer;
        buffer_barrier.offset = 0;
        buffer_barrier.size = VK_WHOLE_SIZE;

        dev.dt.cmdPipelineBarrier(render_cmd,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0,
                                  nullptr, 1, &buffer_barrier, 0, nullptr);

        // Record rendering for this mini batch
        glm::u32vec2 minibatch_offset =
            batch_state.batchFBOffsets[global_batch_offset];
        render_pass_info.renderArea.offset = {
            static_cast<int32_t>(minibatch_offset.x),
            static_cast<int32_t>(minibatch_offset.y),
        };
        render_pass_info.renderArea.extent = {
            per_minibatch_render_size_.x,
            per_minibatch_render_size_.y,
        };

        dev.dt.cmdBeginRenderPass(render_cmd, &render_pass_info,
                                  VK_SUBPASS_CONTENTS_INLINE);

        for (uint32_t local_batch_idx = 0; local_batch_idx < mini_batch_size_;
             local_batch_idx++) {
            uint32_t batch_idx = global_batch_offset + local_batch_idx;
            const Environment &env = envs[batch_idx];
            const VulkanScene &scene =
                *static_cast<const VulkanScene *>(env.getScene().get());

            dev.dt.cmdBindDescriptorSets(
                render_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_.rasterState.drawLayout, 1, 1, &scene.drawSet.hdl,
                0, nullptr);

            glm::u32vec2 batch_offset = batch_state.batchFBOffsets[batch_idx];

            DrawPushConstant draw_const {
                batch_idx,
            };

            dev.dt.cmdPushConstants(
                render_cmd, pipeline_.rasterState.drawLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                sizeof(DrawPushConstant), &draw_const);

            VkViewport viewport;
            viewport.x = batch_offset.x;
            viewport.y = batch_offset.y;
            viewport.width = per_elem_render_size_.x;
            viewport.height = per_elem_render_size_.y;
            viewport.minDepth = 0.f;
            viewport.maxDepth = 1.f;
            dev.dt.cmdSetViewport(render_cmd, 0, 1, &viewport);

            dev.dt.cmdBindIndexBuffer(render_cmd, scene.data.buffer,
                                      scene.indexOffset, VK_INDEX_TYPE_UINT32);

            VkDeviceSize indirect_offset =
                batch_state.indirectBaseOffset +
                batch_state.drawOffsets[batch_idx] *
                    sizeof(VkDrawIndexedIndirectCommand);

            VkDeviceSize count_offset = batch_state.indirectCountBaseOffset +
                                        batch_idx * sizeof(uint32_t);

            dev.dt.cmdDrawIndexedIndirectCountKHR(
                render_cmd, indirect_draw_buffer_.buffer, indirect_offset,
                indirect_draw_buffer_.buffer, count_offset,
                batch_state.maxNumDraws[batch_idx],
                sizeof(VkDrawIndexedIndirectCommand));
        }
        dev.dt.cmdEndRenderPass(render_cmd);

        global_batch_offset += mini_batch_size_;
    }
    REQ_VK(dev.dt.endCommandBuffer(render_cmd));

    render_input_buffer_.flush(dev);

    uint32_t rendered_batch_idx = cur_batch_;

    VkSubmitInfo gfx_submit {VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             nullptr,
                             0,
                             nullptr,
                             nullptr,
                             uint32_t(batch_state.commands.size()),
                             batch_state.commands.data(),
                             0,
                             nullptr};

    graphics_queues_[0].submit(dev, 1, &gfx_submit, batch_state.fence);

    cur_batch_ = (cur_batch_ + 1) & batch_mask_;

    return rendered_batch_idx;
}

void VulkanBackend::waitForFrame(uint32_t batch_idx)
{
    VkFence fence = batch_states_[batch_idx].fence;
    assert(fence != VK_NULL_HANDLE);
    waitForFenceInfinitely(dev, fence);
    resetFence(dev, fence);
}

uint8_t *VulkanBackend::getColorPointer(uint32_t batch_idx)
{
    return (uint8_t *)fb_.extBuffer.getDevicePointer() +
           batch_states_[batch_idx].colorBufferOffset;
}

float *VulkanBackend::getDepthPointer(uint32_t batch_idx)
{
    return (float *)((uint8_t *)fb_.extBuffer.getDevicePointer() +
                     batch_states_[batch_idx].depthBufferOffset);
}

}
}
