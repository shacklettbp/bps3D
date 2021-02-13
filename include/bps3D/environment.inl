#pragma once

#include <glm/gtc/matrix_transform.hpp>

namespace bps3D {

namespace CameraHelper {

static glm::mat4 makePerspectiveMatrix(float hfov,
                                       float aspect,
                                       float near,
                                       float far)
{
    float half_tan = tan(glm::radians(hfov) / 2.f);

    return glm::mat4(1.f / half_tan, 0.f, 0.f, 0.f, 0.f, -aspect / half_tan,
                     0.f, 0.f, 0.f, 0.f, far / (near - far), -1.f, 0.f, 0.f,
                     far * near / (near - far), 0.f);
}

static inline glm::mat4 makeViewMatrix(const glm::vec3 &position,
                                       const glm::vec3 &fwd,
                                       const glm::vec3 &up,
                                       const glm::vec3 &right)
{
    glm::mat4 v(1.f);
    v[0][0] = right.x;
    v[1][0] = right.y;
    v[2][0] = right.z;
    v[0][1] = up.x;
    v[1][1] = up.y;
    v[2][1] = up.z;
    v[0][2] = -fwd.x;
    v[1][2] = -fwd.y;
    v[2][2] = -fwd.z;
    v[3][0] = -glm::dot(right, position);
    v[3][1] = -glm::dot(up, position);
    v[3][2] = glm::dot(fwd, position);

    return v;
}

}

Camera::Camera(const glm::mat4 &world_to_camera,
               float horizontal_fov,
               float aspect_ratio,
               float near,
               float far)
    : worldToCamera(world_to_camera),
      proj(CameraHelper::makePerspectiveMatrix(horizontal_fov,
                                               aspect_ratio,
                                               near,
                                               far))
{}

Camera::Camera(const glm::vec3 &position,
               const glm::vec3 &fwd,
               const glm::vec3 &up,
               const glm::vec3 &right,
               float horizontal_fov,
               float aspect_ratio,
               float near,
               float far)
    : worldToCamera(CameraHelper::makeViewMatrix(position, fwd, up, right)),
      proj(CameraHelper::makePerspectiveMatrix(horizontal_fov,
                                               aspect_ratio,
                                               near,
                                               far))
{}

void Camera::updateView(const glm::mat4 &world_to_camera)
{
    worldToCamera = world_to_camera;
}

void Camera::updateView(const glm::vec3 &position,
                        const glm::vec3 &fwd,
                        const glm::vec3 &up,
                        const glm::vec3 &right)
{
    worldToCamera = CameraHelper::makeViewMatrix(position, fwd, up, right);
}

uint32_t Environment::addInstance(uint32_t model_idx,
                                  uint32_t material_idx,
                                  const glm::mat4x4 &matrix)
{
    return addInstance(model_idx, material_idx, glm::mat4x3(matrix));
}

const glm::mat4x3 &Environment::getInstanceTransform(uint32_t inst_id) const
{
    const auto &p = index_map_[inst_id];
    return transforms_[p.first][p.second];
}

void Environment::updateInstanceTransform(uint32_t inst_id,
                                          const glm::mat4x3 &mat)
{
    const auto &p = index_map_[inst_id];
    transforms_[p.first][p.second] = mat;
}

void Environment::updateInstanceTransform(uint32_t inst_id,
                                          const glm::mat4 &mat)
{
    updateInstanceTransform(inst_id, glm::mat4x3(mat));
}

void Environment::setInstanceMaterial(uint32_t inst_id, uint32_t material_idx)
{
    const auto &p = index_map_[inst_id];
    materials_[p.first][p.second] = material_idx;
}

void Environment::setCameraView(const glm::mat4 &world_to_camera)
{
    camera_.updateView(world_to_camera);
}

void Environment::setCameraView(const glm::vec3 &position,
                                const glm::vec3 &fwd,
                                const glm::vec3 &up,
                                const glm::vec3 &right)
{
    camera_.updateView(position, fwd, up, right);
}

const std::shared_ptr<Scene> Environment::getScene() const
{
    return scene_;
}

const EnvironmentBackend *Environment::getBackend() const
{
    return backend_.getState();
}

const Camera &Environment::getCamera() const
{
    return camera_;
}

const std::vector<std::vector<glm::mat4x3>> &Environment::getTransforms() const
{
    return transforms_;
}

const std::vector<std::vector<uint32_t>> &Environment::getMaterials() const
{
    return materials_;
}

}
