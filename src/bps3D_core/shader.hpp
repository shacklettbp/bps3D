#pragma once

#include <glm/glm.hpp>

namespace bps3D {

namespace Shader {

using glm::vec3;
using glm::vec2;
using uint = uint32_t;
using glm::uvec4;

#include "shader_common.h"

}

using Shader::Vertex;
using Shader::MaterialParams;
using Shader::MeshChunk;

}
