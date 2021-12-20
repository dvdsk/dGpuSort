#include <cassert>
#include <cstdint>
#include <vector>
#include <mpi.h>
#include "dist.hpp"

using std::vector;

unsigned int id()
{
	int id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	assert(id >= 0);
	return static_cast<unsigned int>(id);
}
unsigned int nodes()
{
	int nodes;
	MPI_Comm_size(MPI_COMM_WORLD, &nodes);
	assert(nodes >= 0);
	return static_cast<unsigned int>(nodes);
}

namespace dist
{

void init(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
}

bool main_process() {
	return id() == 0;
}

vector<uint32_t> into_buckets(vector<uint32_t> data) {
	vector<uint32_t> buckets;

	return buckets;
}

void fan_out(vector<uint32_t> data) {
}

void wait_till_done() {
	MPI_Barrier(MPI_COMM_WORLD);
}

vector<uint32_t> fan_in() {
	vector<uint32_t> sorted;
	return sorted;
}

Task recieve() {
	Task task;
	return task;
}

void send(vector<uint32_t> data) {

}

void cleanup() {
    MPI_Finalize();
}

} // namespace dist
