#include "util.hpp"
#include <limits>
#include <random>
#include <stdlib.h>
#include <algorithm>

using std::vector;

std::vector<uint32_t> random_arry(int n, int seed) {

	std::mt19937 rng(seed);

	auto max = std::numeric_limits<uint32_t>::max();
	std::uniform_int_distribution<uint32_t> dist(0, max);
	auto generator = [&dist, &rng](){ return dist(rng); };

	vector<uint32_t> data(n);
	std::generate(data.begin(), data.end(), generator);
		
	// return value optimization will prevent copy
	return data;
}
