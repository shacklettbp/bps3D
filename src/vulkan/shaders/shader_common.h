#ifndef BPS3D_VK_SHADER_COMMON_H_INCLUDED
#define BPS3D_VK_SHADER_COMMON_H_INCLUDED

#include "bps3D_core/shader_common.h"

struct ViewInfo {
    mat4 projection;
    mat4 view;
};

struct DrawPushConstant {
    uint batchIdx;
};

struct PackedLight {
    vec4 position;
    vec4 color;
};

#define MAX_MATERIALS (1000)
#define MAX_LIGHTS (2000)
#define WORKGROUP_SIZE (32)

#endif
