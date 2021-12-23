#include "../util.hpp"
#include "../seq_sort.hpp"
#include <cassert>

using util::Slice;
using seq_sort::quick_sort;

using std::vector;
int main() {
	vector<uint32_t> data = util::random_array(20, 0);

	Slice slice(data, 0, data.size());
	quick_sort(slice);

	auto sorted = slice.as_vec();
	util::assert_sort(sorted, data);
}
