#include "../util.hpp"
#include "../gpu.hpp"

using std::vector;
int main() {
	vector<uint32_t> data = util::random_array(10, 0);
	sort(data);
	// test(data);

	assert(util::is_sorted(data) || "data was not sorted");
}
