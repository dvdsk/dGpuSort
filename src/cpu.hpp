#pragma once

#include <cstdint>
#include <vector>

namespace cpu
{
std::vector<uint32_t> sort(std::vector<uint32_t> &data);
std::vector<uint64_t> offsets(const std::vector<uint32_t>& data, const uint32_t n_buckets);
std::vector<uint32_t> place_elements(const std::vector<uint32_t> &data,
				     const std::vector<uint64_t> &offsets,
				     const uint32_t n_buckets);
} // namespace cpu
