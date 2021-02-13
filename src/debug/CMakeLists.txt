add_library(bps3D_debug SHARED
    ../../include/bps3D/debug.hpp renderdoc_interop.cpp
)

target_include_directories(bps3D_debug
    PUBLIC ${MAIN_INCLUDE_DIR}
)
 target_link_libraries(bps3D_debug renderdoc)
