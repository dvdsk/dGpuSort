#include "../util.hpp"
#include "../cpu.hpp"
#include <cassert>

using std::vector;
int main() {
	vector<uint32_t> data = util::random_array(10, 0);
	cpu::sort(data);

	assert(util::is_sorted(data) || !"data not sorted");
}
