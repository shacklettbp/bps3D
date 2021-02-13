add_library(bps3D_core SHARED
    ${MAIN_INCLUDE_DIR}/bps3D/config.hpp
    ${MAIN_INCLUDE_DIR}/bps3D/utils.hpp
    ${MAIN_INCLUDE_DIR}/bps3D/environment.hpp
    ${MAIN_INCLUDE_DIR}/bps3D/backend.hpp
    scene.hpp scene.cpp
    utils.hpp utils.cpp
    shader.hpp 
)

target_include_directories(bps3D_core
    PUBLIC 
        ${MAIN_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_link_libraries(bps3D_core
    PUBLIC
        CUDA::cudart
        Threads::Threads
        glm
)
