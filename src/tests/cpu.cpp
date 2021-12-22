#include "../util.hpp"
#include "../cpu.hpp"
#include "../seq_sort.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>

using std::vector;
int main(int argc, char *argv[]) {
	unsigned long seed = 0;
	if (argc == 2) {
		seed = std::stoul(argv[1]);
	}

	vector<uint32_t> data = util::random_array(20, seed);
	auto sorted = cpu::sort(data);

	util::assert_sort(sorted, data);
}
