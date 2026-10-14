#pragma once

#include "cnpy.h"
#include <string>

namespace roboparty {
cnpy::npz_t load_npz(const std::string& path);
cnpy::NpyArray load_npz(const std::string& path, const std::string& name);
}
