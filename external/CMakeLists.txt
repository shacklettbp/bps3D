IF (NOT glm_FOUND OR USE_BUNDLED)
    message(STATUS "Using bundled GLM")
    add_library(glm INTERFACE)
    target_include_directories(glm SYSTEM
        INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm")
ENDIF()

set(MESHOPT_BUILD_SHARED_LIBS ON)
add_subdirectory(meshoptimizer EXCLUDE_FROM_ALL)

set(SIMDJSON_JUST_LIBRARY ON CACHE INTERNAL "")
set(SIMDJSON_BUILD_STATIC ON CACHE INTERNAL "")
add_subdirectory(simdjson EXCLUDE_FROM_ALL)

add_library(renderdoc INTERFACE)
target_include_directories(renderdoc INTERFACE extra)

add_library(spirv_reflect SHARED
    SPIRV-Reflect/spirv_reflect.h SPIRV-Reflect/spirv_reflect.c)
target_include_directories(spirv_reflect
    INTERFACE SPIRV-Reflect)

add_subdirectory(glslang EXCLUDE_FROM_ALL)
target_include_directories(glslang SYSTEM
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glslang/glslang/Public>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glslang/StandAlone>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/glslang/SPIRV>)

find_library(ZSTD_LIB zstd REQUIRED)

add_library(ktx SHARED
    KTX-Software/include/ktx.h
    KTX-Software/lib/basis_sgd.h
    KTX-Software/lib/basis_transcode.cpp
    KTX-Software/lib/basisu/transcoder/basisu_file_headers.h
    KTX-Software/lib/basisu/transcoder/basisu_global_selector_cb.h
    KTX-Software/lib/basisu/transcoder/basisu_global_selector_palette.h
    KTX-Software/lib/basisu/transcoder/basisu_transcoder_internal.h
    KTX-Software/lib/basisu/transcoder/basisu_transcoder_uastc.h
    KTX-Software/lib/basisu/transcoder/basisu_transcoder.cpp
    KTX-Software/lib/basisu/transcoder/basisu_transcoder.h
    KTX-Software/lib/basisu/transcoder/basisu.h
    KTX-Software/lib/checkheader.c
    KTX-Software/lib/dfdutils/createdfd.c
    KTX-Software/lib/dfdutils/colourspaces.c
    KTX-Software/lib/dfdutils/dfd.h
    KTX-Software/lib/dfdutils/dfd2vk.inl
    KTX-Software/lib/dfdutils/interpretdfd.c
    KTX-Software/lib/dfdutils/printdfd.c
    KTX-Software/lib/dfdutils/queries.c
    KTX-Software/lib/dfdutils/vk2dfd.c
    KTX-Software/lib/dfdutils/vulkan/vk_platform.h
    KTX-Software/lib/dfdutils/vulkan/vulkan_core.h
    KTX-Software/lib/etcdec.cxx
    KTX-Software/lib/etcunpack.cxx
    KTX-Software/lib/filestream.c
    KTX-Software/lib/filestream.h
    KTX-Software/lib/formatsize.h
    KTX-Software/lib/gl_format.h
    KTX-Software/lib/gl_funcs.c
    KTX-Software/lib/gl_funcs.h
    KTX-Software/lib/glloader.c
    KTX-Software/lib/hashlist.c
    KTX-Software/lib/info.c
    KTX-Software/lib/ktxint.h
    KTX-Software/lib/memstream.c
    KTX-Software/lib/memstream.h
    KTX-Software/lib/stream.h
    KTX-Software/lib/strings.c
    KTX-Software/lib/swap.c
    KTX-Software/lib/texture.c
    KTX-Software/lib/texture.h
    KTX-Software/lib/texture2.c
    KTX-Software/lib/texture2.h
    KTX-Software/lib/uthash.h
    KTX-Software/lib/vk_format.h
    KTX-Software/lib/vkformat_check.c
    KTX-Software/lib/vkformat_enum.h
    KTX-Software/lib/vkformat_str.c
    KTX-shim/shim.c
)

target_include_directories(ktx
    PUBLIC
        KTX-Software/include
    PRIVATE 
        KTX-Software/lib/basisu/transcoder
        KTX-Software/other_include)

target_compile_definitions(ktx PRIVATE LIBKTX)

# Only enable featuers we need
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_FXT1=0)
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_DXT1=0)
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_DXT5A=0)
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_PVRTC1=0)
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_ETC2_EAC_A8=0)
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_ASTC=0)
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_ATC=0)
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_ETC2_EAC_RG11=0)
target_compile_definitions(ktx PRIVATE BASISD_SUPPORT_PVRTC2=0)
target_compile_definitions(ktx PUBLIC KTX_FEATURE_KTX2)
target_compile_definitions(ktx PUBLIC KTX_OMIT_VULKAN=1)

target_link_libraries(ktx PRIVATE
    dl
    Threads::Threads
    ${ZSTD_LIB}
)

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE stb)
