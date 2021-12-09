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

namespace gpu {
vector<uint32_t> sort(vector<uint32_t>& data) {
	vector<uint32_t> buckets;
	return buckets;
}
}
