#pragma once

#include <bps3D/fwd.hpp>
#include <bps3D/backend.hpp>
#include <bps3D/utils.hpp>

#include <glm/glm.hpp>
#include <vector>

namespace bps3D {

struct EnvironmentInit;

struct Camera {
    inline Camera(const glm::mat4 &world_to_camera,
                  float horizontal_fov,
                  float aspect_ratio,
                  float near,
                  float far);

    inline Camera(const glm::vec3 &position,
                  const glm::vec3 &fwd,
                  const glm::vec3 &up,
                  const glm::vec3 &right,
                  float horizontal_fov,
                  float aspect_ratio,
                  float near,
                  float far);

    inline void updateView(const glm::mat4 &world_to_camera);

    inline void updateView(const glm::vec3 &position,
                           const glm::vec3 &fwd,
                           const glm::vec3 &up,
                           const glm::vec3 &right);

    glm::mat4 worldToCamera;
    glm::mat4 proj;
};

class Environment {
public:
    Environment(EnvironmentImpl &&backend,
                const Camera &cam,
                const std::shared_ptr<Scene> &scene);

    Environment(const Environment &) = delete;
    Environment &operator=(const Environment &) = delete;

    Environment(Environment &&) = default;
    Environment &operator=(Environment &&) = default;

    // Instance transformations
    inline uint32_t addInstance(uint32_t model_idx,
                                uint32_t material_idx,
                                const glm::mat4 &model_matrix);

    uint32_t addInstance(uint32_t model_idx,
                         uint32_t material_idx,
                         const glm::mat4x3 &model_matrix);

    void deleteInstance(uint32_t inst_id);

    inline const glm::mat4x3 &getInstanceTransform(uint32_t inst_id) const;

    inline void updateInstanceTransform(uint32_t inst_id,
                                        const glm::mat4 &model_matrix);

    inline void updateInstanceTransform(uint32_t inst_id,
                                        const glm::mat4x3 &model_matrix);

    inline void setInstanceMaterial(uint32_t inst_id, uint32_t material_idx);

    inline void setCameraView(const glm::mat4 &world_to_camera);
    inline void setCameraView(const glm::vec3 &position,
                              const glm::vec3 &fwd,
                              const glm::vec3 &up,
                              const glm::vec3 &right);

    uint32_t addLight(const glm::vec3 &position, const glm::vec3 &color);
    void removeLight(uint32_t light_id);

    inline const std::shared_ptr<Scene> getScene() const;
    inline const EnvironmentBackend *getBackend() const;
    inline const Camera &getCamera() const;

    inline const std::vector<std::vector<glm::mat4x3>> &getTransforms() const;

    inline const std::vector<std::vector<uint32_t>> &getMaterials() const;

private:
    EnvironmentImpl backend_;
    std::shared_ptr<Scene> scene_;

    Camera camera_;

    std::vector<std::vector<glm::mat4x3>> transforms_;
    std::vector<std::vector<uint32_t>> materials_;

    std::vector<std::pair<uint32_t, uint32_t>> index_map_;
    std::vector<std::vector<uint32_t>> reverse_id_map_;
    std::vector<uint32_t> free_ids_;

    std::vector<uint32_t> free_light_ids_;
    std::vector<uint32_t> light_ids_;
    std::vector<uint32_t> light_reverse_ids_;
};

}

#include <bps3D/environment.inl>
