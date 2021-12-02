#include "../util.hpp"
#include "../gpu.hpp"

int main() {
	auto data = util::random_array(10, 0);

	assert(util::is_sorted(data) || "data was not sorted");
}
