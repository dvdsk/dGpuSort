#include "../util.hpp"
#include "../gpu.hpp"
#include <cassert>

using std::vector;
int main() {
	vector<uint32_t> data = util::random_array(10, 5);

	dbg(data);
	auto sorted = gpu::sort(data);
	dbg(sorted);

	util::assert_sort(sorted, data);
}
