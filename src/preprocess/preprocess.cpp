#include <bps3D/preprocess.hpp>
#include <bps3D_core/utils.hpp>

#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/type_precision.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/norm.hpp>
#include <meshoptimizer.h>

#include "import.hpp"

using namespace std;

namespace bps3D {

using namespace SceneImport;

struct PreprocessData {
    SceneDescription<Vertex, Material> desc;
    string textureDir;
};

static PreprocessData parseSceneData(string_view scene_path,
                                     const glm::mat4 &base_txfm,
                                     optional<string_view> texture_dir,
                                     bool dump_textures)
{
    string serialized_tex_dir;
    if (!texture_dir.has_value()) {
        serialized_tex_dir = "./";
    } else {
        serialized_tex_dir = texture_dir.value();
    }
    return PreprocessData {
        SceneDescription<Vertex, Material>::parseScene(
            scene_path, base_txfm,
            dump_textures ? texture_dir : optional<string_view>()),
        serialized_tex_dir,
    };
}

ScenePreprocessor::ScenePreprocessor(string_view gltf_path,
                                     const glm::mat4 &base_txfm,
                                     optional<string_view> texture_dir,
                                     bool dump_textures)
    : scene_data_(new PreprocessData(
          parseSceneData(gltf_path, base_txfm, texture_dir, dump_textures)))
{}

template <typename VertexType>
static vector<uint32_t> filterDegenerateTriangles(
    const vector<VertexType> &vertices,
    const vector<uint32_t> &orig_indices)
{
    vector<uint32_t> new_indices;
    new_indices.reserve(orig_indices.size());

    uint32_t num_indices = orig_indices.size();
    uint32_t tri_align = orig_indices.size() % 3;
    if (tri_align != 0) {
        cerr << "Warning: non multiple of 3 indices in mesh" << endl;
        num_indices -= tri_align;
    }
    assert(orig_indices.size() % 3 == 0);

    auto getPosition = [](const Vertex &v) {
        return glm::vec3(v.px, v.py, v.pz);
    };

    for (uint32_t i = 0; i < num_indices;) {
        uint32_t a_idx = orig_indices[i++];
        uint32_t b_idx = orig_indices[i++];
        uint32_t c_idx = orig_indices[i++];

        glm::vec3 a = getPosition(vertices[a_idx]);
        glm::vec3 b = getPosition(vertices[b_idx]);
        glm::vec3 c = getPosition(vertices[c_idx]);

        glm::vec3 ab = a - b;
        glm::vec3 bc = b - c;
        float check = glm::length2(glm::cross(ab, bc));

        if (check < 1e-20f) {
            continue;
        }

        new_indices.push_back(a_idx);
        new_indices.push_back(b_idx);
        new_indices.push_back(c_idx);
    }

    uint32_t num_degenerate = orig_indices.size() - new_indices.size();

    if (num_degenerate > 0) {
        cout << "Filtered: " << num_degenerate << " degenerate triangles"
             << endl;
    }

    return new_indices;
}

template <typename VertexType>
struct ProcessedMesh : public Mesh<VertexType> {
    vector<MeshChunk> chunks;
};

template <typename VertexType>
vector<MeshChunk> assignChunks(const vector<VertexType> &vertices,
                               const vector<uint32_t> &indices)
{
    static constexpr int num_vertices_per_meshlet = 64;
    static constexpr int num_triangles_per_meshlet = 126;
    static constexpr int num_meshlets_per_chunk = 32;

    vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(
        indices.size(), num_vertices_per_meshlet, num_triangles_per_meshlet));

    uint32_t num_meshlets = meshopt_buildMeshlets(
        meshlets.data(), indices.data(), indices.size(), vertices.size(),
        num_vertices_per_meshlet, num_triangles_per_meshlet);

    meshlets.resize(num_meshlets);

    int num_chunks =
        (num_meshlets + num_meshlets_per_chunk - 1) / num_meshlets_per_chunk;

    vector<MeshChunk> chunks;
    chunks.reserve(num_chunks);

    uint32_t idx_offset = 0;
    int meshlet_offset = 0;
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int num_chunk_meshlets =
            min((int)num_meshlets - meshlet_offset, num_meshlets_per_chunk);

        glm::vec3 center(0.f);
        float radius = 0.f;
        uint32_t num_triangles = 0;

        array<pair<glm::vec3, float>, num_meshlets_per_chunk> meshlet_bounds;

        for (int meshlet_idx = 0; meshlet_idx < num_chunk_meshlets;
             meshlet_idx++) {
            const auto &meshlet = meshlets[meshlet_offset + meshlet_idx];

            meshopt_Bounds bounds = meshopt_computeMeshletBounds(
                &meshlet, &vertices[0].px, vertices.size(),
                sizeof(VertexType));

            assert(bounds.radius != 0);
            auto meshlet_center = glm::make_vec3(bounds.center);

            meshlet_bounds[meshlet_idx] =
                make_pair(meshlet_center, bounds.radius);

            center += meshlet_center / float(num_chunk_meshlets);

            num_triangles += meshlet.triangle_count;
        }

        for (int meshlet_idx = 0; meshlet_idx < num_chunk_meshlets;
             meshlet_idx++) {
            const auto &[meshlet_center, meshlet_radius] =
                meshlet_bounds[meshlet_idx];

            float to_center = glm::distance(meshlet_center, center);

            radius = max(radius, to_center + meshlet_radius);
        }

        chunks.push_back(MeshChunk {
            center,
            radius,
            idx_offset,
            num_triangles,
            {},
        });

        meshlet_offset += num_chunk_meshlets;
        idx_offset += num_triangles * 3;
    }

    return chunks;
}

template <typename VertexType>
optional<ProcessedMesh<VertexType>> processMesh(
    const Mesh<VertexType> &orig_mesh)
{
    const vector<VertexType> &orig_vertices = orig_mesh.vertices;
    const vector<uint32_t> &orig_indices = orig_mesh.indices;

    vector<uint32_t> filtered_indices =
        filterDegenerateTriangles(orig_vertices, orig_indices);

    if (filtered_indices.size() == 0) {
        cerr << "Warning: removing entire degenerate mesh" << endl;
        return optional<ProcessedMesh<VertexType>>();
    }

    uint32_t num_indices = filtered_indices.size();

    vector<uint32_t> index_remap(orig_vertices.size());
    size_t new_vertex_count = meshopt_generateVertexRemap(
        index_remap.data(), filtered_indices.data(), num_indices,
        orig_vertices.data(), orig_vertices.size(), sizeof(VertexType));

    vector<uint32_t> new_indices(num_indices);
    vector<VertexType> new_vertices(new_vertex_count);

    meshopt_remapIndexBuffer(new_indices.data(), filtered_indices.data(),
                             num_indices, index_remap.data());

    meshopt_remapVertexBuffer(new_vertices.data(), orig_vertices.data(),
                              orig_vertices.size(), sizeof(VertexType),
                              index_remap.data());

    meshopt_optimizeVertexCache(new_indices.data(), new_indices.data(),
                                num_indices, new_vertex_count);

    new_vertex_count = meshopt_optimizeVertexFetch(
        new_vertices.data(), new_indices.data(), num_indices,
        new_vertices.data(), new_vertex_count, sizeof(VertexType));
    new_vertices.resize(new_vertex_count);

    auto chunks = assignChunks(new_vertices, new_indices);

    return ProcessedMesh<VertexType> {
        {
            move(new_vertices),
            move(new_indices),
        },
        move(chunks),
    };
}

template <typename VertexType>
struct ProcessedGeometry {
    vector<ProcessedMesh<VertexType>> meshes;
    vector<uint32_t> meshIDRemap;
    vector<MeshInfo> meshInfos;
    uint32_t totalVertices;
    uint32_t totalIndices;
    uint32_t totalChunks;
};

template <typename VertexType, typename MaterialType>
static ProcessedGeometry<VertexType> processGeometry(
    const SceneDescription<VertexType, MaterialType> &desc)
{
    const auto &orig_meshes = desc.meshes;
    vector<ProcessedMesh<VertexType>> processed_meshes;

    vector<uint32_t> mesh_id_remap(orig_meshes.size());

    for (uint32_t mesh_idx = 0; mesh_idx < orig_meshes.size(); mesh_idx++) {
        const auto &orig_mesh = orig_meshes[mesh_idx];
        auto processed = processMesh<VertexType>(orig_mesh);

        if (processed.has_value()) {
            mesh_id_remap[mesh_idx] = processed_meshes.size();

            processed_meshes.emplace_back(move(*processed));
        } else {
            mesh_id_remap[mesh_idx] = ~0U;
        }
    }

    assert(processed_meshes.size() > 0);

    uint32_t num_vertices = 0;
    uint32_t num_indices = 0;
    uint32_t num_chunks = 0;

    vector<MeshInfo> mesh_infos;
    for (auto &mesh : processed_meshes) {
        // Rewrite indices to refer to the global vertex array
        for (uint32_t &idx : mesh.indices) {
            idx += num_vertices;
        }

        // Change all chunk offsets to be global
        for (auto &chunk : mesh.chunks) {
            chunk.indexOffset += num_indices;
        }

        mesh_infos.push_back(MeshInfo {
            num_indices,
            num_chunks,
            uint32_t(mesh.indices.size() / 3),
            uint32_t(mesh.vertices.size()),
            uint32_t(mesh.chunks.size()),
        });

        num_vertices += mesh.vertices.size();
        num_indices += mesh.indices.size();
        num_chunks += mesh.chunks.size();
    }

    return ProcessedGeometry<VertexType> {
        move(processed_meshes), move(mesh_id_remap), move(mesh_infos),
        num_vertices,           num_indices,         num_chunks,
    };
}

static MaterialMetadata stageMaterials(const vector<Material> &materials,
                                       const string &texture_dir)
{
    vector<string> albedo_textures;
    unordered_map<string, size_t> albedo_tracker;

    vector<MaterialParams> params;
    params.reserve(materials.size());

    for (const Material &material : materials) {
        uint32_t albedo_idx = -1;
        if (!material.albedoName.empty()) {
            auto [iter, inserted] = albedo_tracker.emplace(
                material.albedoName, albedo_textures.size());

            if (inserted) {
                albedo_textures.emplace_back(material.albedoName);
            }

            albedo_idx = iter->second;
        }

        params.push_back({
            material.baseAlbedo,
            material.roughness,
            {albedo_idx, 0, 0, 0},
        });
    }

    return {
        {
            texture_dir,
            move(albedo_textures),
        },
        move(params),
    };
}

void ScenePreprocessor::dump(string_view out_path_name)
{
    auto processed_geometry = processGeometry(scene_data_->desc);

    filesystem::path out_path(out_path_name);
    string basename = out_path.filename();
    basename.resize(basename.rfind('.'));

    ofstream out(out_path, ios::binary);
    auto write = [&](auto val) {
        out.write(reinterpret_cast<const char *>(&val), sizeof(decltype(val)));
    };

    // Pad to 256 (maximum uniform / storage buffer alignment requirement)
    auto write_pad = [&]() {
        constexpr size_t align_req = 256;
        static char pad_buffer[align_req] = {0};

        size_t cur_bytes = out.tellp();
        size_t align = cur_bytes % align_req;
        if (align != 0) {
            out.write(pad_buffer, align_req - align);
        }
    };

    auto align_offset = [](size_t offset) { return (offset + 255) & ~255; };

    auto make_staging_header = [&](const auto &geometry,
                                   const MaterialMetadata &material_metadata) {
        constexpr uint64_t vertex_size =
            sizeof(typename decltype(geometry.meshes[0].vertices)::value_type);
        uint64_t vertex_bytes = vertex_size * geometry.totalVertices;
        uint64_t index_bytes = sizeof(uint32_t) * geometry.totalIndices;
        uint64_t chunk_bytes = sizeof(MeshChunk) * geometry.totalChunks;

        StagingHeader hdr;
        hdr.numMeshes = geometry.meshInfos.size();
        hdr.numVertices = geometry.totalVertices;
        hdr.numIndices = geometry.totalIndices;
        hdr.numChunks = geometry.totalChunks;
        hdr.numMaterials = material_metadata.params.size();

        hdr.indexOffset = align_offset(vertex_bytes);
        hdr.chunkOffset = align_offset(hdr.indexOffset + index_bytes);
        hdr.materialOffset = align_offset(hdr.chunkOffset + chunk_bytes);

        hdr.totalBytes =
            hdr.materialOffset + hdr.numMaterials * sizeof(MaterialParams);

        return hdr;
    };

    auto write_staging = [&](const auto &geometry,
                             const MaterialMetadata &materials,
                             const StagingHeader &hdr) {
        write_pad();

        auto stage_beginning = out.tellp();
        // Write all vertices
        for (const auto &mesh : geometry.meshes) {
            constexpr uint64_t vertex_size =
                sizeof(typename decltype(mesh.vertices)::value_type);
            out.write(reinterpret_cast<const char *>(mesh.vertices.data()),
                      vertex_size * mesh.vertices.size());
        }

        write_pad();
        // Write all indices
        for (const auto &mesh : geometry.meshes) {
            out.write(reinterpret_cast<const char *>(mesh.indices.data()),
                      mesh.indices.size() * sizeof(uint32_t));
        }

        write_pad();
        // Write chunks
        for (const auto &mesh : geometry.meshes) {
            out.write(reinterpret_cast<const char *>(mesh.chunks.data()),
                      mesh.chunks.size() * sizeof(MeshChunk));
        }

        write_pad();
        out.write(reinterpret_cast<const char *>(materials.params.data()),
                  materials.params.size() * sizeof(MaterialParams));

        assert(out.tellp() == int64_t(hdr.totalBytes + stage_beginning));
    };

    auto write_lights = [&](const auto &lights) {
        write(uint32_t(lights.size()));
        for (const auto &light : lights) {
            write(light);
        }
    };

    auto write_textures = [&](const MaterialMetadata &metadata) {
        filesystem::path root_dir =
            filesystem::path(out_path_name).parent_path();
        filesystem::path relative_path =
            filesystem::path(metadata.textureInfo.textureDir)
                .lexically_relative(root_dir);

        string relative_path_str = relative_path.string();
        if (relative_path_str != "") {
            relative_path_str += "/";
        }

        out.write(relative_path_str.data(), relative_path_str.size());
        out.put(0);

        write(uint32_t(metadata.textureInfo.albedo.size()));
        for (const auto &src_tex_name : metadata.textureInfo.albedo) {
            static const char ktx_ext[] = ".ktx2";

            out.write(src_tex_name.data(), src_tex_name.rfind('.'));

            // Includes nul terminator
            out.write(ktx_ext, sizeof(ktx_ext));
        }
    };

    auto write_instances = [&](const auto &desc,
                               const vector<uint32_t> &mesh_id_remap) {
        const vector<InstanceProperties> &instances = desc.defaultInstances;
        uint32_t num_instances = instances.size();
        for (const InstanceProperties &orig_inst : instances) {
            if (mesh_id_remap[orig_inst.meshIndex] == ~0U) {
                num_instances--;
            }
        }

        write(uint32_t(num_instances));
        for (const InstanceProperties &orig_inst : instances) {
            uint32_t new_mesh_id = mesh_id_remap[orig_inst.meshIndex];
            if (new_mesh_id == ~0U) continue;

            write(uint32_t(new_mesh_id));
            write(uint32_t(orig_inst.materialIndex));
            write(orig_inst.txfm);
        }
    };

    auto write_scene = [&](const auto &geometry, const auto &desc,
                           const auto &texture_dir) {
        const auto &materials = desc.materials;
        auto material_metadata = stageMaterials(materials, texture_dir);

        StagingHeader hdr = make_staging_header(geometry, material_metadata);
        write(hdr);
        write_pad();

        // Write mesh infos
        out.write(reinterpret_cast<const char *>(geometry.meshInfos.data()),
                  hdr.numMeshes * sizeof(MeshInfo));

        write_lights(desc.defaultLights);

        write_textures(material_metadata);

        write_instances(desc, geometry.meshIDRemap);

        write_staging(geometry, material_metadata, hdr);
    };

    // Header: magic
    write(uint32_t(0x55555555));
    write(uint32_t(SceneLoadData::formatVersion));
    write_scene(processed_geometry, scene_data_->desc,
                scene_data_->textureDir);
    out.close();
}

template struct HandleDeleter<PreprocessData>;

}
