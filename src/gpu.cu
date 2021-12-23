#include "util.hpp"
#include "seq_sort.hpp"
#include <cuda_runtime.h>
#include <vector>

using std::vector;
using std::size_t;
using util::Slice;
using util::div_up;

// adapted from https://stackoverflow.com/a/14038590
#define gpu_assert(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
static void bucket_loop(Slice<uint32_t> data, Slice<uint64_t> offsets, uint32_t bucket_width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i+1 >= offsets.size()) {
		return;
	}

	// inner loop will run in each GPU thread
	uint32_t start_inc = bucket_width * i;
	uint32_t end_excl = bucket_width * (i + 1);
	uint64_t bucket_size = 0;

	for (const auto numb : data) {
		if (numb >= start_inc && numb < end_excl) {
			bucket_size += 1;
		}
	}
	offsets[i+1] = bucket_size;
}

static Slice<uint32_t> data_to_gpu(const vector<uint32_t> &data) {
	uint32_t* d_data = nullptr;
	const auto data_size = data.size()*sizeof(uint32_t);
	auto ok = cudaMalloc((void**) &d_data, data_size);
	gpu_assert(ok);

	const uint32_t* h_data = data.data();
	ok = cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
	gpu_assert(ok);

	Slice d_slice(d_data, data.size());
	return d_slice;
}

struct GpuConfig {
	unsigned int warps;
	/// threads per core
	unsigned int threads;
};

// 2 cuda cores per SMM, 24 SMM for titan x -> 48 cores
// max 1536 threads per core. 512 is more ideal. 32 can simultaniously do work
// more then 32 is better to hide memory latency etc
static GpuConfig gpu_config(unsigned int n_tasks) {
	GpuConfig config;
	/* config.warps = 10;
	config.threads = div_up(n_tasks, config.warps); */
	config.threads = 256;
	config.warps = div_up(n_tasks, config.threads);
	return config;
}

/// count the needed size of each bucket and store them starting
/// at index 1. Set index 0 to 0. After this function the gpu will have
/// the bucket offsets at the pointer returned by this function
static Slice<uint64_t> bucket_sizes(Slice<uint32_t> d_data, const uint32_t n_buckets)
{
	uint64_t* d_arr = nullptr;
	const auto offsets_size = (n_buckets + 1) * sizeof(uint64_t);
	auto ok = cudaMalloc((void**) &d_arr, offsets_size);
	gpu_assert(ok);

	Slice d_sizes(d_arr, n_buckets+1);
	auto bucket_width = std::numeric_limits<uint32_t>::max() / n_buckets;
	auto conf = gpu_config(n_buckets-1);
	bucket_loop<<<conf.warps,conf.threads-1>>>(d_data, d_sizes, bucket_width);
	ok = cudaDeviceSynchronize();
	gpu_assert(ok);

	return d_sizes;
}

__global__
static void d_to_offsets(Slice<uint64_t> d_sizes, uint64_t data_len) {
	auto prev = 0;
	for(auto& s : d_sizes) {
		s += prev;
		prev = s;
	}
	d_sizes[d_sizes.size()-1] = data_len;
	d_sizes[0] = 0;
}

static Slice<uint64_t> to_offsets(Slice<uint64_t> d_sizes, uint64_t data_len) {
	dbg(data_len);
	d_to_offsets<<<1,1>>>(d_sizes, data_len);
	auto ok = cudaDeviceSynchronize();
	gpu_assert(ok);
	return d_sizes;
}

__global__
static void d_place_elements(uint32_t bucket_width, Slice<uint32_t> data, 
		Slice<uint32_t> buckets, Slice<uint64_t> offsets) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= offsets.size()) {
		return;
	}

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

static Slice<uint32_t> place_elements(Slice<uint32_t> d_data, Slice<uint64_t> d_offsets, int n_buckets) {
	uint32_t* d_arr = nullptr;
	const auto buckets_size = d_data.size() * sizeof(uint32_t);
	auto ok = cudaMalloc((void**) &d_arr, buckets_size);
	gpu_assert(ok);

	auto bucket_width = std::numeric_limits<uint32_t>::max() / n_buckets;
	
	Slice d_buckets(d_arr, d_data.size());
	auto conf = gpu_config(n_buckets-1);
	d_place_elements<<<conf.warps, conf.threads>>>(bucket_width, d_data, d_buckets, d_offsets);
	ok = cudaDeviceSynchronize();
	gpu_assert(ok);

	return d_buckets;
}

static vector<uint32_t> download_sorted(Slice<uint32_t> d_buckets) {
	auto sorted = util::filled_vec<uint32_t>(d_buckets.size(), 0);
	auto ok = cudaMemcpy(sorted.data(), d_buckets.start, d_buckets.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	gpu_assert(ok);
	return sorted;
}

__global__
static void d_sort_buckets(Slice<uint32_t> buckets, Slice<uint64_t> offsets, size_t n_buckets) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_buckets) {
		return;
	}

	auto length = offsets[i + 1] - offsets[i];
	Slice bucket(buckets, offsets[i], length);
	seq_sort::quick_sort(bucket);
}

static void sort_buckets(Slice<uint32_t> d_buckets, Slice<uint64_t> d_offsets, size_t n_buckets) {
	auto conf = gpu_config(n_buckets-1);
	dbg(conf.warps, conf.threads);
	d_sort_buckets<<<conf.warps, conf.threads>>>(d_buckets, d_offsets, n_buckets);
	auto ok = cudaDeviceSynchronize();
	gpu_assert(ok);
}

namespace gpu {
vector<uint32_t> sort(vector<uint32_t>& data) {
	auto d_data = data_to_gpu(data);
	auto n = util::n_buckets(data.size(), true);
	dbg(n);
	auto d_sizes = bucket_sizes(d_data, n);
	auto d_offsets = to_offsets(d_sizes, data.size());
	
	auto offsets = util::filled_vec<uint64_t>(d_offsets.size(), 0);
	auto ok = cudaMemcpy(offsets.data(), d_offsets.start, d_offsets.size()*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	gpu_assert(ok);
	dbg(offsets);

	auto d_buckets = place_elements(d_data, d_offsets, n);
	vector<uint32_t> view_buckets = util::filled_vec<uint32_t>(d_buckets.size(), 0);
	dbg(d_buckets.size());
	ok = cudaMemcpy(view_buckets.data(), d_buckets.start, d_buckets.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	gpu_assert(ok);
	dbg(view_buckets);

	sort_buckets(d_buckets, d_offsets, n);

	return download_sorted(d_buckets);
	/* vector<uint32_t> test;
	return test; */
}
}
