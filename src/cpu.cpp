#include "util.hpp"
#include "gpu.hpp"
#include <cstdint>

static void swap(uint32_t& a, uint32_t& b) {
	uint32_t temp = a;
	a = b;
	b = temp;
}

namespace cpu {

void sort(std::vector<uint32_t>& data){
	for (auto& a : data) {
		auto done = true;
		for (auto& b : data) {
			if (a < b) {
				swap(a,b);
				done = false;
			}
		}

		if(done) {
			break;
		}
	}
}

}
