#include <iostream>
#include <cstdlib>
#include <cstring>
#include <glm/gtx/string_cast.hpp>

#include <bps3D/preprocess.hpp>

using namespace std;

int main(int argc, const char *argv[])
{
    if (argc < 3) {
        cerr
            << argv[0]
            << " SRC DST [X_AXIS Y_AXIS Z_AXIS] [TEXTURE_DIR] [--texture-dump]"
            << endl;
        exit(EXIT_FAILURE);
    }

    glm::mat4 base_txfm(1.f);

    if (argc > 3) {
        if (argc < 6) {
            cerr << argv[0] << ": Need to specify zero or all source axes"
                 << endl;
            exit(EXIT_FAILURE);
        }
        const char *x = argv[3];
        const char *y = argv[4];
        const char *z = argv[5];

        auto convertArg = [argv](const char *desc) {
            if (!strcmp(desc, "up")) {
                return glm::vec4(0, 1, 0, 0);
            }
            if (!strcmp(desc, "down")) {
                return glm::vec4(0, -1, 0, 0);
            }
            if (!strcmp(desc, "right")) {
                return glm::vec4(1, 0, 0, 0);
            }
            if (!strcmp(desc, "left")) {
                return glm::vec4(-1, 0, 0, 0);
            }
            if (!strcmp(desc, "forward")) {
                return glm::vec4(0, 0, 1, 0);
            }
            if (!strcmp(desc, "backward")) {
                return glm::vec4(0, 0, -1, 0);
            }
            cerr << argv[0] << ": Invalid axes argument \"" << desc << "\"\n";
            exit(EXIT_FAILURE);
        };

        base_txfm[0] = convertArg(x);
        base_txfm[1] = convertArg(y);
        base_txfm[2] = convertArg(z);
    }

    optional<string_view> texture_dir;
    if (argc > 6) {
        texture_dir.emplace(argv[6]);
    }

    bool dump_textures = false;
    if (argc > 7 && !strcmp(argv[7], "--texture-dump")) {
        dump_textures = true;
    }

    cout << "Transform:\n" << glm::to_string(base_txfm) << endl;

    bps3D::ScenePreprocessor dumper(argv[1], base_txfm, texture_dir,
                                    dump_textures);

    dumper.dump(argv[2]);

    return 0;
}
