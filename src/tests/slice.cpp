#include "../util.hpp"
#include <vector>

using std::vector;
using util::Slice;
using std::numeric_limits;

void assert_slice_val(Slice<int> slice, int val) {
	for (const auto n : slice) {
		assert(n == val);
	}
}

void set_slice_val(Slice<int> slice, int val) {
	for (auto& n : slice) {
		n = val;
	}
}

int main() {
	assert(numeric_limits<std::size_t>::max() == numeric_limits<uint64_t>::max());
	assert(numeric_limits<std::size_t>::min() == numeric_limits<uint64_t>::min());

	vector<int> data(20, 42);
	assert(data.size() == 20);

	Slice a(data, 0*5, 5); 
	Slice b(data, 1*5, 5); 
	Slice c(data, 2*5, 5); 
	Slice d(data, 3*5, 5); 

	set_slice_val(d, 4);
	set_slice_val(a, 1);
	set_slice_val(c, 3);
	set_slice_val(b, 2);

	assert_slice_val(a, 1);
	assert_slice_val(b, 2);
	assert_slice_val(c, 3);
	assert_slice_val(d, 4);
}
