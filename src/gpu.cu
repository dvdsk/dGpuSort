#include "util.hpp"
#include <cuda_runtime.h>
#include <vector>

using std::vector;

__global__
void add_one(uint32_t* data, unsigned int len) {
	for (unsigned int i=0; i<len; i++) {
		data[i] += 1;
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

	add_one<<<1, 1>>>(d_data, data.size());

	ok = cudaDeviceSynchronize();
	gpu_assert(ok);

	ok = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyDeviceToHost);
	gpu_assert(ok);

	cudaFree((void*)h_data);
}
}
