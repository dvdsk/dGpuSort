#include "../util.hpp"
#include "../cpu.hpp"
#include "../gpu.hpp"
#include "../dist.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mpi.h>

using std::vector;
int main(int argc, char **argv)
{
	unsigned int seed = 0;
	dist::init(argc, argv);
	if (dist::main_process()) {
		vector<uint32_t> data = util::random_array(200000, seed);
		auto buckets = dist::into_buckets(data);
		dist::fan_out(buckets);
		dist::wait_till_done();
		auto sorted = dist::fan_in();
		util::assert_sort(sorted, data);
	} else {
		auto task = dist::recieve();
		vector<uint32_t> sorted;
		if (task.use_gpu) {
			sorted = gpu::sort(task.data);
		} else {
			sorted = cpu::sort(task.data);
		}
		dist::send(sorted);
	}
	dist::cleanup();
}
