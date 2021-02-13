#include <bps3D.hpp>
#include <bps3D_core/common.hpp>
#include <bps3D_core/scene.hpp>
#include <bps3D_core/utils.hpp>

#include "vulkan/render.hpp"

#include <functional>
#include <iostream>

using namespace std;

namespace bps3D {

AssetLoader::AssetLoader(LoaderImpl &&backend) : backend_(move(backend))
{}

shared_ptr<Scene> AssetLoader::loadScene(string_view scene_path)
{
    SceneLoadData load_data = SceneLoadData::loadFromDisk(scene_path);

    return backend_.loadScene(move(load_data));
}

static bool enableValidation()
{
    char *enable_env = getenv("BPS3D_VALIDATE");
    if (!enable_env || enable_env[0] == '0') return false;

    return true;
}

static RendererImpl makeBackend(const RenderConfig &cfg, BackendSelect backend)
{
    bool validate = enableValidation();

    switch (backend) {
        case BackendSelect::Vulkan: {
            auto *renderer = new vk::VulkanBackend(cfg, validate);
            return makeRendererImpl<vk::VulkanBackend>(renderer);
        }
    }

    cerr << "Unknown backend" << endl;
    abort();
}

Renderer::Renderer(const RenderConfig &cfg, BackendSelect backend)
    : backend_(makeBackend(cfg, backend)),
      aspect_ratio_(float(cfg.imgWidth) / float(cfg.imgHeight))
{}

AssetLoader Renderer::makeLoader()
{
    return AssetLoader(backend_.makeLoader());
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene)
{
    Camera default_cam(glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f),
                       glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f, 0.f, 0.f),
                       90.f, 1.f, 0.001f, 10000.f);
    return Environment(backend_.makeEnvironment(default_cam, scene),
                       default_cam, scene);
}

Environment Renderer::makeEnvironment(const shared_ptr<Scene> &scene,
                                      const glm::mat4 &world_to_camera,
                                      float horizontal_fov,
                                      float aspect_ratio,
                                      float near,
                                      float far)
{
    Camera cam(world_to_camera, horizontal_fov,
               aspect_ratio == 0.f ? aspect_ratio_ : aspect_ratio, near, far);

    return Environment(backend_.makeEnvironment(cam, scene), cam, scene);
}

Environment Renderer::makeEnvironment(const std::shared_ptr<Scene> &scene,
                                      const glm::vec3 &pos,
                                      const glm::vec3 &fwd,
                                      const glm::vec3 &up,
                                      const glm::vec3 &right,
                                      float horizontal_fov,
                                      float aspect_ratio,
                                      float near,
                                      float far)
{
    Camera cam(pos, fwd, up, right, horizontal_fov,
               aspect_ratio == 0.f ? aspect_ratio_ : aspect_ratio, near, far);

    return Environment(backend_.makeEnvironment(cam, scene), cam, scene);
}

uint32_t Renderer::render(const Environment *envs)
{
    return backend_.render(envs);
}

void Renderer::waitForFrame(uint32_t batch_idx)
{
    backend_.waitForFrame(batch_idx);
}

uint8_t *Renderer::getColorPointer(uint32_t batch_idx)
{
    return backend_.getColorPointer(batch_idx);
}

float *Renderer::getDepthPointer(uint32_t batch_idx)
{
    return backend_.getDepthPointer(batch_idx);
}

Environment::Environment(EnvironmentImpl &&backend,
                         const Camera &cam,
                         const shared_ptr<Scene> &scene)
    : backend_(move(backend)),
      scene_(scene),
      camera_(cam),
      transforms_(scene_->envInit.transforms),
      materials_(scene_->envInit.materials),
      index_map_(scene_->envInit.indexMap),
      reverse_id_map_(scene_->envInit.reverseIDMap),
      free_ids_(),
      free_light_ids_(),
      light_ids_(scene_->envInit.lightIDs),
      light_reverse_ids_(scene_->envInit.lightReverseIDs)
{
    // FIXME use EnvironmentInit lights
}

uint32_t Environment::addInstance(uint32_t model_idx,
                                  uint32_t material_idx,
                                  const glm::mat4x3 &model_matrix)
{
    transforms_[model_idx].emplace_back(model_matrix);
    materials_[model_idx].emplace_back(material_idx);
    uint32_t instance_idx = transforms_[model_idx].size() - 1;

    uint32_t outer_id;
    if (free_ids_.size() > 0) {
        uint32_t free_id = free_ids_.back();
        free_ids_.pop_back();
        index_map_[free_id].first = model_idx;
        index_map_[free_id].second = instance_idx;

        outer_id = free_id;
    } else {
        index_map_.emplace_back(model_idx, instance_idx);
        outer_id = index_map_.size() - 1;
    }

    reverse_id_map_[model_idx].emplace_back(outer_id);

    return outer_id;
}

void Environment::deleteInstance(uint32_t inst_id)
{
    auto [model_idx, instance_idx] = index_map_[inst_id];
    auto &transforms = transforms_[model_idx];
    auto &materials = materials_[model_idx];
    auto &reverse_ids = reverse_id_map_[model_idx];

    if (transforms.size() > 1) {
        // Keep contiguous
        transforms[instance_idx] = transforms.back();
        materials[instance_idx] = materials.back();
        reverse_ids[instance_idx] = reverse_ids.back();
        index_map_[reverse_ids[instance_idx]] = {model_idx, instance_idx};
    }
    transforms.pop_back();
    materials.pop_back();
    reverse_ids.pop_back();

    free_ids_.push_back(inst_id);
}

uint32_t Environment::addLight(const glm::vec3 &position,
                               const glm::vec3 &color)
{
    backend_.addLight(position, color);
    uint32_t light_idx = light_reverse_ids_.size();

    uint32_t light_id;
    if (free_light_ids_.size() > 0) {
        uint32_t free_id = free_light_ids_.back();
        free_light_ids_.pop_back();
        light_ids_[free_id] = light_idx;

        light_id = free_id;
    } else {
        light_ids_.push_back(light_idx);
        light_id = light_ids_.size() - 1;
    }

    light_reverse_ids_.push_back(light_idx);
    return light_id;
}

void Environment::removeLight(uint32_t light_id)
{
    uint32_t light_idx = light_ids_[light_id];
    backend_.removeLight(light_idx);

    if (light_reverse_ids_.size() > 1) {
        light_reverse_ids_[light_idx] = light_reverse_ids_.back();
        light_ids_[light_reverse_ids_[light_idx]] = light_idx;
    }
    light_reverse_ids_.pop_back();

    free_light_ids_.push_back(light_id);
}

EnvironmentImpl::EnvironmentImpl(DestroyType destroy_ptr,
                                 AddLightType add_light_ptr,
                                 RemoveLightType remove_light_ptr,
                                 EnvironmentBackend *state)
    : destroy_ptr_(destroy_ptr),
      add_light_ptr_(add_light_ptr),
      remove_light_ptr_(remove_light_ptr),
      state_(state)
{}

EnvironmentImpl::EnvironmentImpl(EnvironmentImpl &&o)
    : destroy_ptr_(o.destroy_ptr_),
      add_light_ptr_(o.add_light_ptr_),
      remove_light_ptr_(o.remove_light_ptr_),
      state_(o.state_)
{
    o.state_ = nullptr;
}

EnvironmentImpl &EnvironmentImpl::operator=(EnvironmentImpl &&o)
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }

    destroy_ptr_ = o.destroy_ptr_;
    add_light_ptr_ = o.add_light_ptr_;
    remove_light_ptr_ = o.remove_light_ptr_;
    state_ = o.state_;

    o.state_ = nullptr;

    return *this;
}

EnvironmentImpl::~EnvironmentImpl()
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }
}

uint32_t EnvironmentImpl::addLight(const glm::vec3 &position,
                                   const glm::vec3 &color)
{
    return invoke(add_light_ptr_, state_, position, color);
}

void EnvironmentImpl::removeLight(uint32_t idx)
{
    invoke(remove_light_ptr_, state_, idx);
}

LoaderImpl::LoaderImpl(DestroyType destroy_ptr,
                       LoadSceneType load_scene_ptr,
                       LoaderBackend *state)
    : destroy_ptr_(destroy_ptr),
      load_scene_ptr_(load_scene_ptr),
      state_(state)
{}

LoaderImpl::LoaderImpl(LoaderImpl &&o)
    : destroy_ptr_(o.destroy_ptr_),
      load_scene_ptr_(o.load_scene_ptr_),
      state_(o.state_)
{
    o.state_ = nullptr;
}

LoaderImpl::~LoaderImpl()
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }
}

LoaderImpl &LoaderImpl::operator=(LoaderImpl &&o)
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }

    destroy_ptr_ = o.destroy_ptr_;
    load_scene_ptr_ = o.load_scene_ptr_;
    state_ = o.state_;

    o.state_ = nullptr;

    return *this;
}

shared_ptr<Scene> LoaderImpl::loadScene(SceneLoadData &&scene_data)
{
    return invoke(load_scene_ptr_, state_, move(scene_data));
}

RendererImpl::RendererImpl(DestroyType destroy_ptr,
                           MakeLoaderType make_loader_ptr,
                           MakeEnvironmentType make_env_ptr,
                           RenderType render_ptr,
                           WaitType wait_ptr,
                           GetColorType get_color_ptr,
                           GetDepthType get_depth_ptr,
                           RenderBackend *state)
    : destroy_ptr_(destroy_ptr),
      make_loader_ptr_(make_loader_ptr),
      make_env_ptr_(make_env_ptr),
      render_ptr_(render_ptr),
      wait_ptr_(wait_ptr),
      get_color_ptr_(get_color_ptr),
      get_depth_ptr_(get_depth_ptr),
      state_(state)
{}

RendererImpl::RendererImpl(RendererImpl &&o)
    : destroy_ptr_(o.destroy_ptr_),
      make_loader_ptr_(o.make_loader_ptr_),
      make_env_ptr_(o.make_env_ptr_),
      render_ptr_(o.render_ptr_),
      wait_ptr_(o.wait_ptr_),
      get_color_ptr_(o.get_color_ptr_),
      get_depth_ptr_(o.get_depth_ptr_),
      state_(o.state_)
{
    o.state_ = nullptr;
}

RendererImpl &RendererImpl::operator=(RendererImpl &&o)
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }

    destroy_ptr_ = o.destroy_ptr_;
    make_loader_ptr_ = o.make_loader_ptr_;
    make_env_ptr_ = o.make_env_ptr_;
    render_ptr_ = o.render_ptr_;
    wait_ptr_ = o.wait_ptr_;
    get_color_ptr_ = o.get_color_ptr_;
    get_depth_ptr_ = o.get_depth_ptr_;
    state_ = o.state_;

    o.state_ = nullptr;

    return *this;
}

RendererImpl::~RendererImpl()
{
    if (state_) {
        invoke(destroy_ptr_, state_);
    }
}

LoaderImpl RendererImpl::makeLoader()
{
    return invoke(make_loader_ptr_, state_);
}

EnvironmentImpl RendererImpl::makeEnvironment(
    const Camera &cam,
    const std::shared_ptr<Scene> &scene) const
{
    return invoke(make_env_ptr_, state_, cam, scene);
}

uint32_t RendererImpl::render(const Environment *envs)
{
    return invoke(render_ptr_, state_, envs);
}

void RendererImpl::waitForFrame(uint32_t frame_idx)
{
    invoke(wait_ptr_, state_, frame_idx);
}

uint8_t *RendererImpl::getColorPointer(uint32_t frame_idx)
{
    return invoke(get_color_ptr_, state_, frame_idx);
}

float *RendererImpl::getDepthPointer(uint32_t frame_idx)
{
    return invoke(get_depth_ptr_, state_, frame_idx);
}

}
