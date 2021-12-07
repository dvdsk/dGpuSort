#include <cstdint>
#include <vector>

using std::vector;

namespace dist {
	void sort(vector<uint32_t>& data, bool use_gpu) {
		if (use_gpu) {
			sort_gpu(data);
		} else {
			sort_cpu(data);
		}
	}
}
