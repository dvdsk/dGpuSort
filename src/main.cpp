#include "util.hpp"
#include "cpu.hpp"
#include "gpu.hpp"
#include "dist.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

using std::vector;

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

void sort_on_host(util::Slice<uint32_t> slice, bool use_gpu)
{
	vector<uint32_t> data;
	data.reserve(slice.size());
	for (const auto e : slice) {
		data.push_back(e);
	}

	vector<uint32_t> sorted;
	if (use_gpu) {
		sorted = gpu::sort(data);
	} else {
		sorted = cpu::sort(data);
	}

	for (unsigned int i=0; i<slice.size(); i++) {
		slice[i] = sorted[i];
	}
}

using std::chrono::steady_clock;
using std::chrono::duration;
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
		vector<uint32_t> to_sort = util::random_array(size, seed);

		auto start = steady_clock::now();
		auto tasks = dist::to_tasks(to_sort);
		dist::fan_out(tasks);

		sort_on_host(tasks.slices[0], use_gpu);

		dist::fan_in(tasks);
		auto end = steady_clock::now();
		auto elapsed = duration<double>(end-start).count();
		printf("%.20f\n", elapsed);

		auto sorted = std::move(tasks.data);
		dist::wait_till_done();
		dbg("done");
		util::assert_sort(sorted, to_sort);

		// for (unsigned int i=0; i<sorted.size(); i+=10000) {
		// 	printf("%u\n", sorted[i]);
		// }

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
