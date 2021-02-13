#include "scene.hpp"
#include <vulkan/vulkan_core.h>

#include "shader.hpp"
#include "utils.hpp"

#include <ktx.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace bps3D {
namespace vk {

static FrustumBounds computeFrustumBounds(const glm::mat4 &proj)
{
    glm::mat4 t = glm::transpose(proj);
    glm::vec4 xplane = t[3] + t[0];
    glm::vec4 yplane = t[3] + t[1];

    xplane /= glm::length(glm::vec3(xplane));
    yplane /= glm::length(glm::vec3(yplane));

    float znear = proj[3][2] / proj[2][2];
    float zfar = znear * proj[2][2] / (1.f + proj[2][2]);

    return {
        glm::vec4(xplane.x, xplane.z, yplane.y, yplane.z),
        glm::vec2(znear, zfar),
    };
}

VulkanEnvironment::VulkanEnvironment(const Camera &cam,
                                     const VulkanScene &scene)
    : EnvironmentBackend {},
      frustumBounds(computeFrustumBounds(cam.proj)),
      lights()
{
    for (const LightProperties &light : scene.envInit.lights) {
        lights.push_back({
            glm::vec4(light.color, 1.f),
            glm::vec4(light.position, 1.f),
        });
    }
}

uint32_t VulkanEnvironment::addLight(const glm::vec3 &position,
                                     const glm::vec3 &color)
{
    lights.push_back(PackedLight {
        glm::vec4(position, 1.f),
        glm::vec4(color, 1.f),
    });

    return lights.size() - 1;
}

void VulkanEnvironment::removeLight(uint32_t idx)
{
    lights[idx] = lights.back();
    lights.pop_back();
}

struct StagedTexture {
    uint32_t width;
    uint32_t height;
    uint32_t numLevels;

    ktxTexture *data;
};

VulkanLoader::VulkanLoader(const DeviceState &d,
                           MemoryAllocator &alc,
                           const QueueState &transfer_queue,
                           const QueueState &gfx_queue,
                           const ShaderPipeline &cull_shader,
                           const ShaderPipeline &draw_shader,
                           bool need_materials,
                           bool need_lighting)
    : dev(d),
      alloc(alc),
      transfer_queue_(transfer_queue),
      gfx_queue_(gfx_queue),
      transfer_cmd_pool_(makeCmdPool(d, d.transferQF)),
      transfer_stage_cmd_(makeCmdBuffer(dev, transfer_cmd_pool_)),
      gfx_cmd_pool_(makeCmdPool(d, d.gfxQF)),
      gfx_copy_cmd_(makeCmdBuffer(dev, gfx_cmd_pool_)),
      ownership_sema_(makeBinarySemaphore(dev)),
      fence_(makeFence(dev)),
      cull_desc_mgr_(dev, cull_shader, 1),
      draw_desc_mgr_(dev, draw_shader, 1),
      need_materials_(need_materials),
      need_lighting_(need_lighting)
{}

static void ktxCheck(KTX_error_code res)
{
    static const char *ktx_errors[] = {
        "KTX_SUCCESS",
        "KTX_FILE_DATA_ERROR",
        "KTX_FILE_ISPIPE",
        "KTX_FILE_OPEN_FAILED",
        "KTX_FILE_OVERFLOW",
        "KTX_FILE_READ_ERROR",
        "KTX_FILE_SEEK_ERROR",
        "KTX_FILE_UNEXPECTED_EOF",
        "KTX_FILE_WRITE_ERROR",
        "KTX_GL_ERROR",
        "KTX_INVALID_OPERATION",
        "KTX_INVALID_VALUE",
        "KTX_NOT_FOUND",
        "KTX_OUT_OF_MEMORY",
        "KTX_TRANSCODE_FAILED",
        "KTX_UNKNOWN_FILE_FORMAT",
        "KTX_UNSUPPORTED_TEXTURE_TYPE",
        "KTX_UNSUPPORTED_FEATURE",
        "KTX_LIBRARY_NOT_LINKED",
    };

    if (res != KTX_SUCCESS) {
        const char *ktx_error;
        if (res <= KTX_LIBRARY_NOT_LINKED) {
            ktx_error = ktx_errors[res];
        } else {
            ktx_error = "unknown error";
        }
        cerr << "Loading failed: failed to load ktx texture - " << ktx_error
             << std::endl;
        fatalExit();
    }
}

static StagedTexture loadKTXFile(FILE *texture_file)
{
    ktxTexture *ktx_texture;

    KTX_error_code result = ktxTexture_CreateFromStdioStream(
        texture_file, KTX_TEXTURE_CREATE_NO_FLAGS, &ktx_texture);

    ktxCheck(result);

    if (ktx_texture->generateMipmaps) {
        cerr << "Only textures with pregenerated mipmaps are supported"
             << endl;
        fatalExit();
    }

    return StagedTexture {
        ktx_texture->baseWidth,
        ktx_texture->baseHeight,
        ktx_texture->numLevels,
        ktx_texture,
    };
}

TextureData::TextureData(const DeviceState &d, MemoryAllocator &a)
    : dev(d),
      alloc(a),
      memory(VK_NULL_HANDLE),
      textures(),
      views()
{}

TextureData::TextureData(TextureData &&o)
    : dev(o.dev),
      alloc(o.alloc),
      memory(o.memory),
      textures(move(o.textures)),
      views(move(o.views))
{
    o.memory = VK_NULL_HANDLE;
}

TextureData::~TextureData()
{
    if (memory == VK_NULL_HANDLE) return;

    for (auto view : views) {
        dev.dt.destroyImageView(dev.hdl, view, nullptr);
    }

    for (auto &texture : textures) {
        alloc.destroyTexture(move(texture));
    }

    dev.dt.freeMemory(dev.hdl, memory, nullptr);
}

shared_ptr<Scene> VulkanLoader::loadScene(SceneLoadData &&load_info)
{
    TextureData texture_store(dev, alloc);

    vector<StagedTexture> cpu_textures;
    vector<LocalTexture> &gpu_textures = texture_store.textures;
    vector<VkImageView> &texture_views = texture_store.views;
    vector<VkDeviceSize> texture_offsets;
    optional<VkDeviceMemory> texture_memory;
    optional<HostBuffer> texture_staging;
    uint32_t num_textures = 0;

    if (need_materials_) {
        cpu_textures.reserve(load_info.textureInfo.albedo.size());
        for (const auto &albedo_name : load_info.textureInfo.albedo) {
            string full_path = load_info.textureInfo.textureDir + albedo_name;

            // Hack to keep file descriptor count down
            FILE *file = fopen(full_path.c_str(), "rb");
            if (!file) {
                cerr << "Texture loading failed: Could not open " << full_path
                     << endl;
                fatalExit();
            }

            auto texture = loadKTXFile(file);
            ktxTexture *ktx = texture.data;
            assert(ktx->classId == ktxTexture2_c);

            ktxTexture2 *ktx2 = reinterpret_cast<ktxTexture2 *>(ktx);
            KTX_error_code res =
                ktxTexture2_TranscodeBasis(ktx2, KTX_TTF_BC7_RGBA, 0);
            ktxCheck(res);
            fclose(file);

            cpu_textures.emplace_back(move(texture));
        }
        num_textures = cpu_textures.size();

        gpu_textures.reserve(num_textures);
        texture_views.reserve(num_textures);
        texture_offsets.reserve(num_textures);

        VkDeviceSize cpu_texture_bytes = 0;
        VkDeviceSize gpu_texture_bytes = 0;
        for (const StagedTexture &texture : cpu_textures) {
            ktxTexture *ktx = texture.data;

            for (uint32_t level = 0; level < texture.numLevels; level++) {
                cpu_texture_bytes += ktxTexture_GetImageSize(ktx, level);
            }

            auto [gpu_texture, reqs] = alloc.makeTexture(
                texture.width, texture.height, texture.numLevels);

            gpu_texture_bytes = alignOffset(gpu_texture_bytes, reqs.alignment);

            texture_offsets.emplace_back(gpu_texture_bytes);
            gpu_textures.emplace_back(move(gpu_texture));

            gpu_texture_bytes += reqs.size;
        }

        if (num_textures > 0) {
            texture_memory = alloc.alloc(gpu_texture_bytes);

            if (!texture_memory.has_value()) {
                cerr << "Out of memory, failed to allocate texture storage"
                     << endl;
                fatalExit();
            }
            texture_store.memory = texture_memory.value();

            texture_staging.emplace(
                alloc.makeStagingBuffer(cpu_texture_bytes));
        }
    }

    // Copy all geometry into single buffer
    optional<LocalBuffer> data_opt =
        alloc.makeLocalBuffer(load_info.hdr.totalBytes);

    if (!data_opt.has_value()) {
        cerr << "Out of memory, failed to allocate geometry storage" << endl;
        fatalExit();
    }

    LocalBuffer data = move(data_opt.value());

    HostBuffer data_staging =
        alloc.makeStagingBuffer(load_info.hdr.totalBytes);

    if (holds_alternative<ifstream>(load_info.data)) {
        ifstream &file = *get_if<ifstream>(&load_info.data);
        file.read((char *)data_staging.ptr, load_info.hdr.totalBytes);
    } else {
        char *data_src = get_if<vector<char>>(&load_info.data)->data();
        memcpy(data_staging.ptr, data_src, load_info.hdr.totalBytes);
    }

    // Bind image memory and create views
    for (uint32_t i = 0; i < num_textures; i++) {
        LocalTexture &gpu_texture = gpu_textures[i];
        VkDeviceSize offset = texture_offsets[i];

        REQ_VK(dev.dt.bindImageMemory(dev.hdl, gpu_texture.image,
                                      texture_store.memory, offset));

        VkImageViewCreateInfo view_info;
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.pNext = nullptr;
        view_info.flags = 0;
        view_info.image = gpu_texture.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = alloc.getFormats().texture;
        view_info.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                                VK_COMPONENT_SWIZZLE_B,
                                VK_COMPONENT_SWIZZLE_A};
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                      gpu_texture.mipLevels, 0, 1};

        VkImageView view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));

        texture_views.push_back(view);
    }

    // Start recording for transfer queue
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(transfer_stage_cmd_, &begin_info));

    // Copy vertex/index buffer onto GPU
    VkBufferCopy copy_settings {};
    copy_settings.size = load_info.hdr.totalBytes;
    dev.dt.cmdCopyBuffer(transfer_stage_cmd_, data_staging.buffer, data.buffer,
                         1, &copy_settings);

    // Set initial texture layouts
    DynArray<VkImageMemoryBarrier> texture_barriers(num_textures);
    for (size_t i = 0; i < num_textures; i++) {
        const LocalTexture &gpu_texture = gpu_textures[i];
        VkImageMemoryBarrier &barrier = texture_barriers[i];

        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.pNext = nullptr;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = gpu_texture.image;
        barrier.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, gpu_texture.mipLevels, 0, 1,
        };
    }

    if (num_textures > 0) {
        dev.dt.cmdPipelineBarrier(
            transfer_stage_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
            texture_barriers.size(), texture_barriers.data());

        // Copy all textures into staging buffer & record cpu -> gpu copies
        uint8_t *base_texture_staging =
            reinterpret_cast<uint8_t *>(texture_staging->ptr);
        VkDeviceSize cur_staging_offset = 0;

        vector<VkBufferImageCopy> copy_infos;
        for (size_t i = 0; i < num_textures; i++) {
            const StagedTexture &cpu_texture = cpu_textures[i];
            const LocalTexture &gpu_texture = gpu_textures[i];
            uint32_t base_width = cpu_texture.width;
            uint32_t base_height = cpu_texture.height;
            uint32_t num_levels = cpu_texture.numLevels;

            copy_infos.resize(num_levels);
            ktxTexture *ktx = cpu_texture.data;
            const uint8_t *ktx_data = ktxTexture_GetData(ktx);

            for (uint32_t level = 0; level < num_levels; level++) {
                // Copy to staging
                VkDeviceSize ktx_level_offset;
                KTX_error_code res = ktxTexture_GetImageOffset(
                    ktx, level, 0, 0, &ktx_level_offset);
                ktxCheck(res);

                VkDeviceSize num_level_bytes =
                    ktxTexture_GetImageSize(ktx, level);

                memcpy(base_texture_staging + cur_staging_offset,
                       ktx_data + ktx_level_offset, num_level_bytes);

                uint32_t level_div = 1 << level;
                uint32_t level_width = max(1U, base_width / level_div);
                uint32_t level_height = max(1U, base_height / level_div);

                // Set level copy
                VkBufferImageCopy copy_info {};
                copy_info.bufferOffset = cur_staging_offset;
                copy_info.imageSubresource.aspectMask =
                    VK_IMAGE_ASPECT_COLOR_BIT;
                copy_info.imageSubresource.mipLevel = level;
                copy_info.imageSubresource.baseArrayLayer = 0;
                copy_info.imageSubresource.layerCount = 1;
                copy_info.imageExtent = {
                    level_width,
                    level_height,
                    1,
                };

                copy_infos[level] = copy_info;

                cur_staging_offset += num_level_bytes;
            }

            // Note that number of copy commands is num_levels
            // not copy_infos.size(), because the vector is shared
            // between textures to avoid allocs
            dev.dt.cmdCopyBufferToImage(
                transfer_stage_cmd_, texture_staging->buffer,
                gpu_texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                num_levels, copy_infos.data());
        }

        // Flush staging buffer
        texture_staging->flush(dev);

        // Transfer queue relinquish texture barriers
        for (VkImageMemoryBarrier &barrier : texture_barriers) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }
    }

    // Transfer queue relinquish geometry
    VkBufferMemoryBarrier geometry_barrier;
    geometry_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    geometry_barrier.pNext = nullptr;
    geometry_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    geometry_barrier.dstAccessMask = 0;
    geometry_barrier.srcQueueFamilyIndex = dev.transferQF;
    geometry_barrier.dstQueueFamilyIndex = dev.gfxQF;

    geometry_barrier.buffer = data.buffer;
    geometry_barrier.offset = 0;
    geometry_barrier.size = load_info.hdr.totalBytes;

    // Geometry & texture barrier execute.
    dev.dt.cmdPipelineBarrier(
        transfer_stage_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &geometry_barrier,
        texture_barriers.size(), texture_barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(transfer_stage_cmd_));

    VkSubmitInfo copy_submit {};
    copy_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    copy_submit.waitSemaphoreCount = 0;
    copy_submit.pWaitSemaphores = nullptr;
    copy_submit.pWaitDstStageMask = nullptr;
    copy_submit.commandBufferCount = 1;
    copy_submit.pCommandBuffers = &transfer_stage_cmd_;
    copy_submit.signalSemaphoreCount = 1;
    copy_submit.pSignalSemaphores = &ownership_sema_;

    transfer_queue_.submit(dev, 1, &copy_submit, VK_NULL_HANDLE);

    // Start recording for graphics queue
    REQ_VK(dev.dt.beginCommandBuffer(gfx_copy_cmd_, &begin_info));

    // Finish moving geometry onto graphics queue family
    // geometry and textures need separate barriers due to different
    // dependent stages
    geometry_barrier.srcAccessMask = 0;
    geometry_barrier.dstAccessMask =
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;

    VkPipelineStageFlags dst_geo_gfx_stage =
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;

    dev.dt.cmdPipelineBarrier(gfx_copy_cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              dst_geo_gfx_stage, 0, 0, nullptr, 1,
                              &geometry_barrier, 0, nullptr);

    if (num_textures > 0) {
        for (VkImageMemoryBarrier &barrier : texture_barriers) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = dev.transferQF;
            barrier.dstQueueFamilyIndex = dev.gfxQF;
        }

        // Finish acquiring mip level 0 on graphics queue and transition layout
        dev.dt.cmdPipelineBarrier(
            gfx_copy_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
            texture_barriers.size(), texture_barriers.data());
    }

    REQ_VK(dev.dt.endCommandBuffer(gfx_copy_cmd_));

    VkSubmitInfo gfx_submit {};
    gfx_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    gfx_submit.waitSemaphoreCount = 1;
    gfx_submit.pWaitSemaphores = &ownership_sema_;
    VkPipelineStageFlags sema_wait_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    gfx_submit.pWaitDstStageMask = &sema_wait_mask;
    gfx_submit.commandBufferCount = 1;
    gfx_submit.pCommandBuffers = &gfx_copy_cmd_;

    gfx_queue_.submit(dev, 1, &gfx_submit, fence_);

    waitForFenceInfinitely(dev, fence_);
    resetFence(dev, fence_);

    for (StagedTexture &tex : cpu_textures) {
        ktxTexture_Destroy(tex.data);
    }

    assert(load_info.textureInfo.albedo.size() <= VulkanConfig::max_materials);

    DescriptorSet cull_set = cull_desc_mgr_.makeSet();
    DescriptorSet draw_set = draw_desc_mgr_.makeSet();

    DescriptorUpdates desc_updates(4);

    // Cull Set Layout
    // 0: Mesh Chunks

    VkDescriptorBufferInfo chunk_buffer_info {
        data.buffer,
        load_info.hdr.chunkOffset,
        load_info.hdr.numChunks * sizeof(MeshChunk),
    };
    desc_updates.storage(cull_set.hdl, &chunk_buffer_info, 0);

    // Draw Set Layout
    // 0: Vertex buffer
    // 1: sampler
    // 2: textures
    // 3: material params

    VkDescriptorBufferInfo vertex_buffer_info {
        data.buffer,
        0,
        load_info.hdr.numVertices * sizeof(Vertex),
    };
    desc_updates.storage(draw_set.hdl, &vertex_buffer_info, 0);

    vector<VkDescriptorImageInfo> descriptor_views;
    descriptor_views.reserve(num_textures);

    VkDescriptorBufferInfo material_buffer_info;
    if (need_materials_) {
        for (int albedo_idx = 0; albedo_idx < (int)num_textures;
             albedo_idx++) {
            VkImageView view = texture_views[albedo_idx];
            descriptor_views.push_back({
                VK_NULL_HANDLE,  // Immutable
                view,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
        }

        desc_updates.textures(draw_set.hdl, descriptor_views.data(),
                              num_textures, 2);

        material_buffer_info.buffer = data.buffer;
        material_buffer_info.offset = load_info.hdr.materialOffset;
        material_buffer_info.range =
            load_info.hdr.numMaterials * sizeof(MaterialParams);
        desc_updates.storage(draw_set.hdl, &material_buffer_info, 3);
    }

    desc_updates.update(dev);

    uint32_t num_meshes = load_info.meshInfo.size();

    return make_shared<VulkanScene>(VulkanScene {
        {
            move(load_info.meshInfo),
            move(load_info.envInit),
        },
        move(texture_store),
        move(cull_set),
        move(draw_set),
        move(data),
        load_info.hdr.indexOffset,
        num_meshes,
    });
}

}
}
