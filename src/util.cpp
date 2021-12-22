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
vector<T> filled_vec(uint64_t len, T value)
{
	vector<T> vec;
	vec.reserve(len);
	for (uint64_t i = 0; i < len; i++) {
		vec.push_back(value);
	}
	return vec;
}
// NVCC wont instanciate templates for cpu code so this needs
// to be done explicitly
template vector<uint32_t> filled_vec<uint32_t>(uint64_t len, uint32_t value);
template vector<uint64_t> filled_vec<uint64_t>(uint64_t len, uint64_t value);

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
static bool is_sorted(vector<uint32_t> &data)
{
	uint32_t prev = 0;
	for (const auto &n : data) {
		if (prev <= n) {
			prev = n;
			continue;
		}
		dbg("prev was smaller then current", prev, n);
		return false;
	}
	return true;
}

void assert_sort(std::vector<uint32_t> &sorted, std::vector<uint32_t> &data)
{
	assert(sorted.size() == data.size() || !"elements went missing");
	assert(util::is_sorted(sorted) || !"data not sorted");
}

static uint32_t divide_round_up(uint32_t n, uint32_t d)
{
	return (n + (d - 1)) / d;
}

uint32_t n_buckets(std::size_t n_elem)
{
	if (n_elem <= 20) {
		return 2; // test case
	}

	constexpr int BUCKET_SIZE = 1024;
	// constexpr int BUCKET_SIZE = 1000;
	return divide_round_up(n_elem, BUCKET_SIZE);
}

int devide_up(int num, int denum)
{
	return (num + denum - 1) / denum;
}
} // namespace util
