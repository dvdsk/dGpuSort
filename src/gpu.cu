#include "util.hpp"
#include <cuda_runtime.h>
#include <vector>

using std::vector;

__global__
void add_one(uint32_t* data, unsigned int len) {
	printf("len:%d\n", len);
	for (unsigned int i=0; i<len; i++) {
		printf("data[%d]: %u\n", i, (unsigned int)data[i]);
		data[i] += 1;
	}
}

__device__
void swap(uint32_t& a, uint32_t& b) {
	uint32_t temp = a;
	a = b;
	b = temp;
}

__global__
void gpu_sort(uint32_t* data, unsigned int len) {
	for (int j=0; j < len-1; j++) { // TODO Sync all blocks if possible
		if(j%2 == 0) { // trusting the optimizer to optimized branch out
			for (int i=0; i <= len/2 - 1; i++) { // TODO uses stride (block + threads)
				uint32_t& a = data[2*i];
				uint32_t& b = data[2*i+1];
				if (a > b) {
					swap(a, b);
				}
			}
		} else {
			for (int i=0; i <= len/2 - 2; i++) {
				uint32_t& a = data[2*i+1];
				uint32_t& b = data[2*i+2];
				if (a > b) {
					swap(a, b);
				}
			}
		}
	}
}

// adapted from https://stackoverflow.com/a/14038590
#define gpu_assert(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace gpu {
void sort(vector<uint32_t>& data) {
	const auto bytes = data.size()*sizeof(uint32_t);
	uint32_t* d_data = nullptr;
	const uint32_t* h_data = data.data();
	auto ok = cudaMalloc((void**) &d_data, bytes);
	gpu_assert(ok);

	ok = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
	gpu_assert(ok);

	/* add_one<<<1, 1>>>(d_data, data.size()); */
	gpu_sort<<<1, 1>>>(d_data, data.size());

	ok = cudaDeviceSynchronize();
	gpu_assert(ok);

	ok = cudaMemcpy((void*)h_data, d_data, bytes, cudaMemcpyDeviceToHost);
	gpu_assert(ok);

	cudaFree((void*)h_data);
}
}
