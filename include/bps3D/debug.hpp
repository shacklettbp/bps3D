#pragma once

#include <cstdint>

namespace bps3D {

class RenderDoc {
public:
    RenderDoc();

    inline void startFrame() const
    {
        if (rdoc_impl_) startImpl();
    }

    inline void endFrame() const
    {
        if (rdoc_impl_) endImpl();
    }

    bool loaded() const { return !!rdoc_impl_; }

private:
    void startImpl() const;
    void endImpl() const;

    void *rdoc_impl_;
};

}
