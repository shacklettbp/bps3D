add_library(bps3D_preprocess SHARED
    gltf.hpp gltf.inl
    habitat_json.hpp habitat_json.inl
    import.hpp import.cpp
    ../../include/bps3D/preprocess.hpp preprocess.cpp
)

target_link_libraries(bps3D_preprocess
    PRIVATE
        bps3D_core
        meshoptimizer
        simdjson
    PUBLIC
        glm
)

target_include_directories(bps3D_preprocess
    INTERFACE
        ${MAIN_INCLUDE_DIR}
)
