#include "util.hpp"
#include "cpu.hpp"
#include "gpu.hpp"
#include "dist.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

static void print_help()
{
	std::cerr
		<< "Expects three arguments: seed (int), number of integers in "
		   "dataset (int) and weather or not to use the gpu (int) where 0 "
		   "is false and 1 is true.";
}

static void parse_args(char *argv[], unsigned long int &seed,
		       unsigned long int &size, bool &use_gpu)
{
	seed = std::stoul(argv[1]);
	size = std::stoul(argv[2]);
	auto gpu_int = std::stoul(argv[3]);
	switch (gpu_int) {
	case 0:
		use_gpu = false;
		break;
	case 1:
		use_gpu = true;
		break;
	default:
		throw std::invalid_argument("use_gpu is either '0' or '1'");
	}
}

using std::vector;
int main(int argc, char *argv[])
{
	if (argc < 3) {
		print_help();
	}

	unsigned long int seed;
	unsigned long int size;
	bool use_gpu;

	try {
		parse_args(argv, seed, size, use_gpu);
	} catch (...) {
		print_help();
		return -1;
	}

	dist::init(argc, argv);
	if (dist::main_process()) {
		vector<uint32_t> data = util::random_array(size, seed);
		auto tasks = dist::to_tasks(data);
		dist::fan_out(tasks);
		dist::fan_in(tasks);
		auto sorted = std::move(tasks.data);
		dist::wait_till_done();
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
		dist::signal_done();
	}
	dist::cleanup();
}
