#include "util.hpp"
#include <cstdint>
#include <limits>
#include <random>
#include <stdlib.h>
#include <algorithm>

using std::vector;
namespace util
{

template <typename T>
// returns a vector filled with zeros
vector<T> filled_vec(uint64_t len)
{
	vector<T> vec;
	vec.reserve(len);
	for (uint64_t i = 0; i < len; i++) {
		vec.push_back(0);
	}
	return vec;
}
// NVCC wont instanciate templates for cpu code so this needs
// to be done explicitly
template vector<uint32_t> filled_vec<uint32_t>(uint64_t len);
template vector<uint64_t> filled_vec<uint64_t>(uint64_t len);

vector<uint32_t> random_array(long unsigned int n, long unsigned int seed)
{
	std::mt19937 rng(seed);

	auto max = std::numeric_limits<uint32_t>::max();
	std::uniform_int_distribution<uint32_t> dist(0, max);
	auto generator = [&dist, &rng]() { return dist(rng); };

	vector<uint32_t> data(n);
	std::generate(data.begin(), data.end(), generator);

	// return value optimization will prevent copy
	return data;
}

// checks if data in array is sorted in ascending order
bool is_sorted(vector<uint32_t> &data)
{
	uint32_t prev = 0;
	for (const auto &n : data) {
		if (prev < n) {
			prev = n;
			continue;
		}
		return false;
	}
	return true;
}
} // namespace util
