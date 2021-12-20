#include "../util.hpp"
#include "../cpu.hpp"
#include "../gpu.hpp"
#include "../dist.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <mpi.h>
#pragma GCC diagnostic pop 

using std::vector;
int main(int argc, char **argv)
{
	bool use_gpu = false;
	unsigned int seed = 0;
	dist::init(argc, argv);
	if (dist::main_process()) {
		vector<uint32_t> data = util::random_array(200000, seed);
		auto tasks = dist::to_tasks(data);
		dist::fan_out(tasks);
		dist::wait_till_done();
		dist::fan_in(tasks);
		auto sorted = std::move(tasks.data);
		util::assert_sort(sorted, data);
	} else {
		auto data = dist::recieve();
		vector<uint32_t> sorted;
		if (use_gpu) {
			sorted = gpu::sort(data);
		} else {
			sorted = cpu::sort(data);
		}
		dist::send(sorted);
	}
	dist::cleanup();
}
