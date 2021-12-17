#include "util.hpp"
#include <cstdint>
#include <vector>

namespace seq_sort {
CUDA_CALLABLE void quick_sort(util::Slice<uint32_t> slice); 
}
