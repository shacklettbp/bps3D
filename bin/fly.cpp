#include <bps3D.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <chrono>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;
using namespace bps3D;

const float mouse_speed = 2e-4;
const float movement_speed = 1.5;
const float rotate_speed = 1.25;

static GLFWwindow *makeWindow(const glm::u32vec2 &dim)
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 0);

    auto window = glfwCreateWindow(dim.x, dim.y, "bps3D", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        cerr << "Failed to create window" << endl;
        abort();
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    return window;
}

struct CameraState {
    glm::vec3 eye;
    glm::vec3 fwd;
    glm::vec3 up;
    glm::vec3 right;
};

static glm::i8vec3 key_movement(0, 0, 0);

void windowKeyHandler(GLFWwindow *window, int key, int, int action, int)
{
    if (action == GLFW_REPEAT) return;

    glm::i8vec3 cur_movement(0, 0, 0);
    switch (key) {
        case GLFW_KEY_ESCAPE: {
            if (action == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
            break;
        }
        case GLFW_KEY_ENTER: {
            if (action == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            break;
        }
        case GLFW_KEY_W: {
            cur_movement.y += 1;
            break;
        }
        case GLFW_KEY_A: {
            cur_movement.x -= 1;
            break;
        }
        case GLFW_KEY_S: {
            cur_movement.y -= 1;
            break;
        }
        case GLFW_KEY_D: {
            cur_movement.x += 1;
            break;
        }
        case GLFW_KEY_Q: {
            cur_movement.z -= 1;
            break;
        }
        case GLFW_KEY_E: {
            cur_movement.z += 1;
            break;
        }
    }

    if (action == GLFW_PRESS) {
        key_movement += cur_movement;
    } else {
        key_movement -= cur_movement;
    }
}

static glm::vec2 cursorPosition(GLFWwindow *window)
{
    double mouse_x, mouse_y;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);

    return glm::vec2(mouse_x, -mouse_y);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        cerr << argv[0] << " scene [verbose]" << endl;
        exit(EXIT_FAILURE);
    }

    if (!glfwInit()) {
        cerr << "GLFW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    glm::u32vec2 img_dims(1920, 1080);

    GLFWwindow *window = makeWindow(img_dims);
    if (glewInit() != GLEW_OK) {
        cerr << "GLEW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    array<GLuint, 2> read_fbos;
    glCreateFramebuffers(2, read_fbos.data());

    array<GLuint, 2> render_textures;
    glCreateTextures(GL_TEXTURE_2D, 2, render_textures.data());

    bool show_camera = false;
    if (argc > 2) {
        if (!strcmp(argv[2], "--cam")) {
            show_camera = true;
        }
    }

    Renderer renderer(
        {0, 1, 1, img_dims.x, img_dims.y, true, RenderMode::UnlitRGB});

    array<cudaStream_t, 2> copy_streams;

    array<cudaGraphicsResource_t, 2> dst_imgs;

    cudaError_t res;
    for (int i = 0; i < 2; i++) {
        glTextureStorage2D(render_textures[i], 1, GL_RGBA8, img_dims.x,
                           img_dims.y);

        res = cudaStreamCreate(&copy_streams[i]);
        if (res != cudaSuccess) {
            cerr << "CUDA stream initialization failed" << endl;
            abort();
        }

        res = cudaGraphicsGLRegisterImage(&dst_imgs[i], render_textures[i],
                                          GL_TEXTURE_2D,
                                          cudaGraphicsMapFlagsWriteDiscard);
        if (res != cudaSuccess) {
            cerr << "Failed to map texture into CUDA" << endl;
            abort();
        }
    }

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    CameraState cam {
        glm::vec3(0, 0, 0),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 1, 0),
        glm::vec3(1, 0, 0),
    };
    glm::vec2 mouse_prev = cursorPosition(window);

    vector<Environment> envs;
    envs.emplace_back(renderer.makeEnvironment(scene, cam.eye, cam.fwd, cam.up,
                                               cam.right, 60.f));

    glfwSetKeyCallback(window, windowKeyHandler);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    uint32_t prev_frame = renderer.render(envs.data());

    auto time_prev = chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
        auto time_cur = chrono::steady_clock::now();
        chrono::duration<float> elapsed_duration = time_cur - time_prev;
        time_prev = time_cur;
        float time_delta = elapsed_duration.count();

        glfwPollEvents();
        glm::vec2 mouse_delta;
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
            glm::vec2 mouse_cur = cursorPosition(window);
            mouse_delta = mouse_cur - mouse_prev;
            mouse_prev = mouse_cur;
        } else {
            mouse_delta = glm::vec2(0, 0);
            mouse_prev = cursorPosition(window);
        }

        cam.right = glm::cross(cam.fwd, cam.up);
        glm::mat3 around_right(
            glm::angleAxis(mouse_delta.y * mouse_speed, cam.right));

        cam.up = around_right * cam.up;

        glm::mat3 around_up(
            glm::angleAxis(-mouse_delta.x * mouse_speed, cam.up));

        cam.fwd = around_up * around_right * cam.fwd;

        glm::mat3 around_fwd(glm::angleAxis(
            float(key_movement.z) * rotate_speed * time_delta, cam.fwd));
        cam.up = around_fwd * cam.up;
        cam.right = around_fwd * around_up * cam.right;

        glm::vec2 movement = movement_speed * time_delta *
                             glm::vec2(key_movement.x, key_movement.y);
        cam.eye += cam.right * movement.x + cam.fwd * movement.y;

        cam.fwd = glm::normalize(cam.fwd);
        cam.up = glm::normalize(cam.up);
        cam.right = glm::normalize(cam.right);

        envs[0].setCameraView(cam.eye, cam.fwd, cam.up, cam.right);
        if (show_camera) {
            cout << "E: " << glm::to_string(cam.eye) << "\n"
                 << "F: " << glm::to_string(cam.fwd) << "\n"
                 << "U: " << glm::to_string(cam.up) << "\n"
                 << "R: " << glm::to_string(cam.right) << "\n";
        }

        uint32_t new_frame = renderer.render(envs.data());
        renderer.waitForFrame(prev_frame);

        uint8_t *output = renderer.getColorPointer(prev_frame);

        glNamedFramebufferTexture(read_fbos[prev_frame], GL_COLOR_ATTACHMENT0,
                                  0, 0);

        res = cudaGraphicsMapResources(1, &dst_imgs[prev_frame],
                                       copy_streams[prev_frame]);
        if (res != cudaSuccess) {
            cerr << "Failed to map opengl resource" << endl;
            abort();
        }

        cudaArray_t dst_arr;
        res = cudaGraphicsSubResourceGetMappedArray(
            &dst_arr, dst_imgs[prev_frame], 0, 0);
        if (res != cudaSuccess) {
            cerr << "Failed to get cuda array from opengl" << endl;
            abort();
        }

        res = cudaMemcpy2DToArrayAsync(
            dst_arr, 0, 0, output, img_dims.x * sizeof(uint8_t) * 4,
            img_dims.x * sizeof(uint8_t) * 4, img_dims.y,
            cudaMemcpyDeviceToDevice, copy_streams[prev_frame]);

        if (res != cudaSuccess) {
            cerr << "buffer to image copy failed " << endl;
        }

        // Seems like it shouldn't be necessary but bad tearing otherwise
        cudaStreamSynchronize(copy_streams[prev_frame]);

        res = cudaGraphicsUnmapResources(1, &dst_imgs[prev_frame],
                                         copy_streams[prev_frame]);
        if (res != cudaSuccess) {
            cerr << "Failed to unmap opengl resource" << endl;
            abort();
        }

        glNamedFramebufferTexture(read_fbos[prev_frame], GL_COLOR_ATTACHMENT0,
                                  render_textures[prev_frame], 0);

        glBlitNamedFramebuffer(read_fbos[prev_frame], 0, 0, img_dims.y,
                               img_dims.x, 0, 0, 0, img_dims.x, img_dims.y,
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glfwSwapBuffers(window);

        prev_frame = new_frame;
    }

    cudaGraphicsUnregisterResource(dst_imgs[0]);
    cudaGraphicsUnregisterResource(dst_imgs[1]);

    cudaStreamDestroy(copy_streams[0]);
    cudaStreamDestroy(copy_streams[1]);

    glDeleteTextures(2, render_textures.data());
    glDeleteFramebuffers(2, read_fbos.data());

    glfwDestroyWindow(window);
}
