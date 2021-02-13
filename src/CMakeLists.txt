add_definitions(-DGLM_FORCE_DEPTH_ZERO_TO_ONE)
SET(MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

add_subdirectory(bps3D_core)

add_subdirectory(debug)
add_subdirectory(preprocess)

# Backends
add_subdirectory(vulkan)

add_library(bps3D SHARED
    ../include/bps3D.hpp bps3D.cpp 
)

target_link_libraries(bps3D
    PRIVATE
        bps3D_core
        bps3D_vulkan
    INTERFACE
        CUDA::cudart
        glm
)

target_include_directories(bps3D
    INTERFACE
        ${MAIN_INCLUDE_DIR}
)
