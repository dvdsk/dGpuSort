#include "util.hpp"
#include "gpu.hpp"

int main() {
	auto data = random_arry(10);

	assert(is_sorted(data) || "data was not sorted")
}
