#include "util.hpp"
#include <limits>
#include <random>
#include <stdlib.h>
#include <algorithm>

using std::vector;
namespace util {

vector<uint32_t> random_array(long unsigned int n, long unsigned int seed) {

	std::mt19937 rng(seed);

	auto max = std::numeric_limits<uint32_t>::max();
	std::uniform_int_distribution<uint32_t> dist(0, max);
	auto generator = [&dist, &rng](){ return dist(rng); };

	vector<uint32_t> data(n);
	std::generate(data.begin(), data.end(), generator);
		
	// return value optimization will prevent copy
	return data;
}

// checks if data in array is sorted in ascending order
bool is_sorted(vector<uint32_t>& data) {
	uint32_t prev = 0;
	for(const auto& n : data) {
		if (n < prev) {
			return false;
		}
	}
	return true;
}
}
