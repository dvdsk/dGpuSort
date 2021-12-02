#pragma once
#include "dbg.h" // compile without warnings enabled
#include <cassert>
#include <cstdint>
#include <vector>

namespace util {

std::vector<uint32_t> random_array(long unsigned int n, long unsigned int seed);

bool is_sorted(std::vector<uint32_t>& data);

}
