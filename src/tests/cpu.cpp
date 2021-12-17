#include "../util.hpp"
#include "../cpu.hpp"
#include "../seq_sort.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>

using std::vector;
int main() {
	vector<uint32_t> data = util::random_array(6, 0);
	dbg(data);
	// auto sorted = cpu::sort(data);
	util::Slice<uint32_t> test(data, 0, data.size());
	// for (std::size_t i=0; i<test.size(); i++) {
	// 	dbg(test[i]);
	// }

	seq_sort::quick_sort(test);
	dbg(data);
	util::assert_sort(data, data);
}
