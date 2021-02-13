#ifndef BPS3D_SHADER_COMMON_H_INCLUDED
#define BPS3D_SHADER_COMMON_H_INCLUDED

struct Vertex {
    float px;
    float py;
    float pz;
    float nx;
    float ny;
    float nz;
    float ux;
    float uy;
};

struct MaterialParams {
    vec3 baseAlbedo;
    float roughness;
    uvec4 texIdxs;
};

struct MeshChunk {
    vec3 center;
    float radius;
    uint indexOffset;
    uint numTriangles;
    uint pad[2];
};

#endif
