#include "util.hpp"
#include <cuda_runtime.h>
#include <vector>

using std::vector;

__device__
void swap(uint32_t& a, uint32_t& b) {
	uint32_t temp = a;
	a = b;
	b = temp;
}

__global__
void gpu_sort(uint32_t* data, unsigned int len, int offset) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i <= len/2 - 1 - offset; i+=stride) {
		uint32_t& a = data[2*i+offset];
		uint32_t& b = data[2*i+offset+1];
		if (a > b) {
			swap(a, b);
		}
	}
}

// adapted from https://stackoverflow.com/a/14038590
#define gpu_assert(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void bubble_sort(vector<uint32_t>& data) {
	const auto bytes = data.size()*sizeof(uint32_t);
	uint32_t* d_data = nullptr;
	const uint32_t* h_data = data.data();
	auto ok = cudaMalloc((void**) &d_data, bytes);
	gpu_assert(ok);

	ok = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
	gpu_assert(ok);

	// TitanX tuning: 24 SMM -> blocks multiple of 24
	auto stride = 5;
	auto blocks = data.size() / stride;

	for (int j=0; j < data.size()-1; j++) {
		if(j%2 == 0) { // trusting the optimizer to optimized branch out
			gpu_sort<<<blocks, 512>>>(d_data, data.size(), 0);
		} else {
			gpu_sort<<<blocks, 512>>>(d_data, data.size(), 1);
		}
		// using CPU Synchronization to ensure all blocks are done before we
		// continue to the next run
		ok = cudaDeviceSynchronize();
		gpu_assert(ok);
	}

	ok = cudaDeviceSynchronize();
	gpu_assert(ok);

	ok = cudaMemcpy((void*)h_data, d_data, bytes, cudaMemcpyDeviceToHost);
	gpu_assert(ok);

	cudaFree((void*)h_data);
}

__global__
void bucket_loop(util::Slice<uint32_t> data, util::Slice<uint64_t> offsets, uint32_t bucket_width) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
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

/// count the needed size of each bucket and store them starting
/// at index 1. Set index 0 to 0. After this function the gpu will have
/// the bucket offsets at the pointer returned by this function
uint64_t* bucket_size(const vector<uint32_t> &data,
				    const uint32_t n_buckets)
{
	uint32_t* d_data = nullptr;
	const auto data_size = data.size()*sizeof(uint32_t);
	auto ok = cudaMalloc((void**) &d_data, data_size);
	gpu_assert(ok);

	uint64_t* d_offsets = nullptr;
	const auto offsets_size = (n_buckets + 1) * sizeof(uint64_t);
	ok = cudaMalloc((void**) &d_offsets, offsets_size);
	gpu_assert(ok);

	const uint32_t* h_data = data.data();
	ok = cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
	gpu_assert(ok);

	uint64_t first_offset_val = 0;
	ok = cudaMemcpy(d_offsets, &first_offset_val, 1*sizeof(uint64_t), cudaMemcpyHostToDevice);
	gpu_assert(ok);
	uint64_t last_offset_val = data.size();
	uint64_t* d_last_offset = d_offsets + offsets_size - sizeof(uint64_t);
	ok = cudaMemcpy(d_last_offset, &last_offset_val, 1*sizeof(uint64_t), cudaMemcpyHostToDevice);
	gpu_assert(ok);

	auto bucket_width = std::numeric_limits<uint32_t>::max() / n_buckets;
	auto s1 = util::Slice(d_data, data.size());

	/* for (uint32_t i = 0; i < n_buckets - 1; i++) { */
	bucket_loop<<<1,n_buckets-1>>>(s1, 
			util::Slice(d_offsets, n_buckets+1), bucket_width);

	return d_offsets;
}

namespace gpu {
vector<uint32_t> sort(vector<uint32_t>& data) {
	auto offsets = util::filled_vec<uint64_t>(2+1);
	auto d_offsets = bucket_size(data, 2);
	auto ok = cudaMemcpy((void*)d_offsets, offsets.data(), 3*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	gpu_assert(ok);
	dbg(offsets);

	/* auto buckets = place_elements(data, offsets, 2); */

	vector<uint32_t> buckets;
	return buckets;
}
}
