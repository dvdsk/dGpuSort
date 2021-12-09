#include "../util.hpp"
#include "../gpu.hpp"
#include <cassert>

using std::vector;
int main() {
	vector<uint32_t> data = util::random_array(10, 0);

	dbg(data);
	auto sorted = gpu::sort(data);
	dbg(data);

	assert(util::is_sorted(sorted) || !"data not sorted");
}
