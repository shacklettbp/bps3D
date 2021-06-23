#pragma once

#include <bps3D_core/shader.hpp>
#include <bps3D_core/scene.hpp>

#include <optional>
#include <vector>

namespace bps3D {
namespace SceneImport {

template <typename VertexType>
struct Mesh {
    std::vector<VertexType> vertices;
    std::vector<uint32_t> indices;
};

struct Material {
    std::string albedoName;
    glm::vec3 baseAlbedo;
    float roughness;

    static Material make(const std::string_view albedo_name,
                         const glm::vec3 &color,
                         float roughness);
};

template <typename VertexType, typename MaterialType>
struct SceneDescription {
    std::vector<Mesh<VertexType>> meshes;
    std::vector<MaterialType> materials;

    std::vector<InstanceProperties> defaultInstances;
    std::vector<LightProperties> defaultLights;

    static SceneDescription parseScene(
        std::string_view scene_path,
        const glm::mat4 &base_txfm,
        std::optional<std::string_view> texture_dir);
};

}
}
