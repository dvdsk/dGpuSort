#include "cpu.hpp"
#include "seq_sort.hpp"
#include "util.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

using std::vector;

// returns a vector with the size of each bucket.
// The sizes start at element 1. This way vector starting from
// element 0 will read the offset to access each bucket if the buckets are
// stored continues.
static vector<uint64_t> bucket_size(const vector<uint32_t> &data,
				    const uint32_t n_buckets)
{
	vector<uint64_t> buckets = { 0 };
	auto bucket_width = std::numeric_limits<uint32_t>::max() / n_buckets;

	// do not measure the size of the last bucket, as it can be inferred
	for (uint32_t i = 0; i < n_buckets; i++) {
		// inner loop will run in each GPU thread
		uint32_t start_inc = bucket_width * i;
		uint32_t end_excl = bucket_width * (i + 1);
		uint64_t bucket_size = 0;

		for (const auto numb : data) {
			if (numb >= start_inc && numb < end_excl) {
				bucket_size += 1;
			}
		}
		buckets.push_back(bucket_size);
	}
	return buckets;
}

static vector<uint64_t> to_offsets(vector<uint64_t> sizes) {
	uint64_t prev = 0;
	for (auto& s : sizes) {
		s += prev;
		prev = s;
	}
	return sizes;
}

static vector<uint32_t> place_elements(const vector<uint32_t> &data,
				       const vector<uint64_t> &offsets,
				       const uint32_t n_buckets)
{
	auto bucket_width = std::numeric_limits<uint32_t>::max() / n_buckets;
	auto buckets = util::filled_vec<uint32_t>(data.size(), 0);

	for (uint32_t i = 0; i < n_buckets; i++) {
		// in parallel over all GPU threads
		uint32_t start_inc = bucket_width * i;
		uint32_t end_excl = bucket_width * (i + 1);

		auto offset = offsets[i];
		for (uint64_t j = 0; j < data.size(); j++) {
			auto numb = data[j];
			if (numb >= start_inc && numb < end_excl) {
				buckets[offset] = numb;
				offset += 1;
			}
		}
	}
	return buckets;
}

static void sort_buckets(vector<uint32_t> &buckets,
			 const vector<uint64_t> &offsets,
			 const uint32_t n_buckets)
{
	for (unsigned int i = 0; i < n_buckets; i++) {
		auto length = offsets[i+1] - offsets[i];
		util::Slice bucket(buckets, offsets[i], length);
		seq_sort::quick_sort(bucket);
	}
}

namespace cpu
{
vector<uint32_t> sort(std::vector<uint32_t> &data)
{
	auto n = util::n_buckets(data.size());
	auto sizes = bucket_size(data, n);
	auto offsets = to_offsets(std::move(sizes));
	// TODO merge and split buckets that are too large/small
	// only if absolutely needed (do not think so, should be uniform?)
	auto buckets = place_elements(data, offsets, n);
	sort_buckets(buckets, offsets, n);
	return buckets;
}
} // namespace cpu
