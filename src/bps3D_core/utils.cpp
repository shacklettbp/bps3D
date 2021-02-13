#include "utils.hpp"
#include <cstdlib>

using namespace std;

namespace bps3D {

[[noreturn]] void fatalExit() noexcept
{
    abort();
}

}
