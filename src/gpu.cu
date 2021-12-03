#include "util.hpp"
#include <cuda_runtime.h>
#include <vector>

using std::vector;

void test(int hi) {
};

// namespace gpu {
void sort(vector<uint32_t>& data) {
	auto bytes = data.size()*sizeof(uint32_t);
	uint32_t* d_data = nullptr;
	uint32_t* h_data = data.data();
	auto alloc_ok = cudaMalloc((void**) &d_data, bytes);
	assert(alloc_ok==true || "could not allocate memory on gpu");
	auto copy_ok = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
	assert(copy_ok==true || "could not copy data to gpu");

	cudaFree(h_data);
}
// }
