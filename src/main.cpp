#include "util.hpp"
#include "dist.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

static void print_help() {
  std::cerr << "Expects three arguments: seed (int), number of integers in "
               "dataset (int) and weather or not to use the gpu (int) where 0 "
               "is false and 1 is true.";
}

static void parse_args(char *argv[], unsigned long int &seed,
                       unsigned long int &size, bool &use_gpu) {
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
int main(int argc, char *argv[]) {
  if (argc != 3) {
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

  vector<uint32_t> data = util::random_array(size, seed);
  dist::sort(data, use_gpu);

  assert(util::is_sorted(data) || !"data not sorted");
}
