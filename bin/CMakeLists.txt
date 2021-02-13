add_executable(preprocess
    preprocess.cpp
)
target_link_libraries(preprocess bps3D_preprocess)

add_executable(singlebench
    singlebench.cpp
)
target_link_libraries(singlebench bps3D)

add_executable(save_frame
    save_frame.cpp
)
target_link_libraries(save_frame bps3D stb bps3D_debug)

find_package(glfw3 3.3 QUIET)
find_package(GLEW QUIET)

if (glfw3_FOUND AND GLEW_FOUND)
    add_executable(fly
        fly.cpp
    )

    target_link_libraries(fly bps3D glfw GLEW GL)
endif()
