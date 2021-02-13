#pragma once

#include <bps3D/fwd.hpp>
#include <bps3D/config.hpp>
#include <bps3D/backend.hpp>
#include <bps3D/utils.hpp>
#include <bps3D/environment.hpp>

#include <string_view>

namespace bps3D {

class AssetLoader {
public:
    AssetLoader(LoaderImpl &&backend);

    std::shared_ptr<Scene> loadScene(std::string_view scene_path);

private:
    LoaderImpl backend_;

    friend class BatchRenderer;
};

class Renderer {
public:
    Renderer(const RenderConfig &cfg,
             BackendSelect backend = BackendSelect::Vulkan);

    AssetLoader makeLoader();

    Environment makeEnvironment(const std::shared_ptr<Scene> &scene);
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                const glm::mat4 &world_to_camera,
                                float horizontal_fov = 90.f,
                                float aspect_ratio = 0.f,
                                float near = 0.01f,
                                float far = 1000.f);
    Environment makeEnvironment(const std::shared_ptr<Scene> &scene,
                                const glm::vec3 &pos,
                                const glm::vec3 &fwd,
                                const glm::vec3 &up,
                                const glm::vec3 &right,
                                float horizontal_fov = 90.f,
                                float aspect_ratio = 0.f,
                                float near = 0.01f,
                                float far = 1000.f);

    uint32_t render(const Environment *envs);

    void waitForFrame(uint32_t batch_idx = 0);

    uint8_t *getColorPointer(uint32_t batch_idx = 0);
    float *getDepthPointer(uint32_t batch_idx = 0);

private:
    RendererImpl backend_;
    float aspect_ratio_;
};

}
