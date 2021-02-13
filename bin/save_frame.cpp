#include <bps3D.hpp>
#include <bps3D/debug.hpp>

#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#include <glm/gtx/transform.hpp>

// FIXME
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace std;
using namespace bps3D;

template <typename T>
static vector<T> copyToHost(const T *dev_ptr,
                            uint32_t width,
                            uint32_t height,
                            uint32_t num_channels)
{
    uint64_t num_pixels = width * height * num_channels;

    vector<T> buffer(num_pixels);

    cudaMemcpy(buffer.data(), dev_ptr, sizeof(T) * num_pixels,
               cudaMemcpyDeviceToHost);

    return buffer;
}

void saveFrame(const char *fname,
               const float *dev_ptr,
               uint32_t width,
               uint32_t height,
               uint32_t num_channels)
{
    auto buffer = copyToHost(dev_ptr, width, height, num_channels);

    vector<uint8_t> sdr_buffer(buffer.size());
    for (unsigned i = 0; i < buffer.size(); i++) {
        float v = buffer[i];
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        sdr_buffer[i] = v * 255;
    }

    stbi_write_bmp(fname, width, height, num_channels, sdr_buffer.data());
}

void saveFrame(const char *fname,
               const uint8_t *dev_ptr,
               uint32_t width,
               uint32_t height,
               uint32_t num_channels)
{
    auto buffer = copyToHost(dev_ptr, width, height, num_channels);

    stbi_write_bmp(fname, width, height, num_channels, buffer.data());
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        cerr << argv[0] << "scene batch_size" << endl;
        exit(EXIT_FAILURE);
    }

    RenderDoc rdoc {};

    uint32_t batch_size = atoi(argv[2]);

    glm::u32vec2 out_dim(256, 256);

    Renderer renderer({0, 1, batch_size, out_dim.x, out_dim.y, false,
                       RenderMode::Depth | RenderMode::UnlitRGB});

    rdoc.startFrame();
    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);
    vector<Environment> envs;

    glm::mat4 base(glm::inverse(glm::mat4(-1.19209e-07, 0, 1, 0, 0, 1, 0, 0,
                                          -1, 0, -1.19209e-07, 0, -3.38921,
                                          1.62114, -3.34509, 1)));

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        glm::mat4 r =
            glm::rotate(glm::radians(10.f * batch_idx), glm::vec3(0, 1, 0));

        glm::mat4 view = r * base;

        envs.emplace_back(
            renderer.makeEnvironment(scene, view, 90.f, 0.f, 0.01, 1000.f));
    }

    renderer.render(envs.data());
    renderer.waitForFrame();

    rdoc.endFrame();

    uint8_t *base_color_ptr = renderer.getColorPointer();
    float *base_depth_ptr = renderer.getDepthPointer();

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        saveFrame(("/tmp/out_color_" + to_string(batch_idx) + ".bmp").c_str(),
                  base_color_ptr + batch_idx * out_dim.x * out_dim.y * 4,
                  out_dim.x, out_dim.y, 4);
        saveFrame(("/tmp/out_depth_" + to_string(batch_idx) + ".bmp").c_str(),
                  base_depth_ptr + batch_idx * out_dim.x * out_dim.y,
                  out_dim.x, out_dim.y, 1);
    }
}
