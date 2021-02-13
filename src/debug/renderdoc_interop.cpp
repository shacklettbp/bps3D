#include <bps3D/debug.hpp>

#include <cassert>
#include <dlfcn.h>
#include <renderdoc/renderdoc.h>

namespace bps3D {

using RenderDocApi = const RENDERDOC_API_1_4_1 *;

static void *initRDoc()
{
    void *lib = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (lib) {
        auto get_api = (pRENDERDOC_GetAPI)dlsym(lib, "RENDERDOC_GetAPI");

        void *ptrs;
        [[maybe_unused]] int ret =
            get_api(eRENDERDOC_API_Version_1_4_1, (void **)&ptrs);
        assert(ret == 1);

        return ptrs;
    } else {
        return nullptr;
    }
}

RenderDoc::RenderDoc() : rdoc_impl_(initRDoc())
{}

void RenderDoc::startImpl() const
{
    ((RenderDocApi)rdoc_impl_)->StartFrameCapture(nullptr, nullptr);
}

void RenderDoc::endImpl() const
{
    ((RenderDocApi)rdoc_impl_)->EndFrameCapture(nullptr, nullptr);
}

}
