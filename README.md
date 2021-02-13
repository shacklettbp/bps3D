bps3D: Batch Rendering for Reinforcement Learning
=================================================

This repository is the reference implementation of batch rendering, as described in _Large Batch Simulation for Deep Reinforcement Learning_.

The implementation is relatively scene and task agnostic: both depth sensor and RGB camera outputs are supported for scenes in the [GLTF](https://www.khronos.org/gltf/) format. Additionally, basic dynamic shading support is available for synthetic scenes.

bps3D can achieve 30,000 to 100,000 FPS on the Gibson and Matterport3D datasets when rendering 64x64 pixel frames.


Dependencies
------------

* CMake
* NVIDIA GPU
* CUDA 10.1 or higher
* NVIDIA driver with Vulkan 1.1 support (440.33 or later confirmed to work)
* Vulkan headers and loader (described below)

The easiest way to obtain the Vulkan dependencies is by installing the latest version of the official Vulkan SDK, available at <https://vulkan.lunarg.com/sdk/home>. Detailed instructions for installing the SDK are [here](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html), specifically the "System Requirements" and "Set up the runtime environment" sections. Be sure that the Vulkan SDK is enabled in your current shell before building.

Building
--------

When using bps3D as the renderer for another CMake-based project, the best integration method is to include this repository as a git submodule, then use CMake's `add_subdirectory` on the submodule directory. Targets that depend on the renderer should link against the `bps3D` target. An example of this integration is in the [bps-nav repository](https://github.com/shacklettbp/bps-nav).

A standalone copy of bps3D can be built as follows:
```bash
git clone --recursive https://github.com/shacklettbp/bps3D
cd bps3D
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
```

Tools and examples will be built in `bps3D/build/bin/`

Scene Preprocessing
-------------------

bps3D requires scenes to be converted into a custom format before rendering. The repository includes a tool for this purpose in `bin/preprocess.cpp` that can be used as follows:

```bash
./build/bin/preprocess src_file.glb dst_file.bps right up forward texture_dir --texture-dump
```

This command will convert a source asset, `src_file.glb`, into bps3D's custom format: `dst_file.bps`. The next three arguments (`right`, `up`, `forward`) describe how the coordinate axes of the source asset are transformed. Finally, `texture_dir` specifies that `dst_file.bps` will be written to expect textures to be in the `texture_dir` directory. The optional `--texture-dump` argument tells `preprocess` to also extract any textures that are embedded in `src_file.glb` to `texture_dir`.

The only currently supported input format is GLTF, although asset importing is decoupled from the overall project and can easily be extended. Specific instructions for converting the Gibson and Matterport3D datasets are included in the [bps-nav repository](https://github.com/shacklettbp/bps-nav).

Texture Compression
-------------------

bps3D relies on texture compression to reduce memory footprint. All textures must be stored in the [KTX2](https://github.khronos.org/KTX-Specification/) container, with UASTC compression. Use the `toktx` tool from the [KTX-Software repository](https://github.com/KhronosGroup/KTX-Software), which has built-in support for common source image formats.

The following is an example command that compresses a JPEG source texture, with rate distortion optimization to reduce on-disk file size:
```bash
toktx --uastc --uastc_rdo_q 0.9 --uastc_rdo_d 1024 --zcmp 20 --genmipmap dst.ktx2 src.jpg
```

Additional Tools
----------------

In addition to the `preprocess` tool, a handful of other tools are available in the `bin/` directory:

* `fly`: A 3D fly camera for `*.bps` files. Depends on OpenGL and GLFW3.
* `singlebench`: Tests renderer performance on a single scene.
* `save_frame`: Test program that renders a batch of RGB and depth outputs for a fixed camera position.

Citation
--------

```
@article{shacklett21bps,
    title   = {Large Batch Simulation for Deep Reinforcement Learning},
    author  = {Brennan Shacklett and Erik Wijmans and Aleksei Petrenko and Manolis Savva and Dhruv Batra and Vladlen Koltun and Kayvon Fatahalian},
    journal = {International Conference On Learning Representations (ICLR)},
    year    = {2021}
}
```
