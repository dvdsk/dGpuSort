#include "../util.hpp"
#include "../gpu.hpp"
#include <cassert>

using std::vector;
int main() {
	vector<uint32_t> data = util::random_array(10, 0);
	gpu::sort(data);

	assert(util::is_sorted(data) || !"data not sorted");
}
