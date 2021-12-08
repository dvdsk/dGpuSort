#include "../util.hpp"
#include "../gpu.hpp"
#include <cassert>

using std::vector;
int main() {
	vector<uint32_t> data = util::random_array(200000, 0);

	dbg(data);
	gpu::sort(data);
	dbg(data);

	assert(util::is_sorted(data) || !"data not sorted");
}
