#include <cassert>
#include <cstdint>
#include <vector>
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <mpi.h>
#pragma GCC diagnostic pop 
#include "util.hpp"
#include "cpu.hpp"
#include "dist.hpp"

using std::vector;

namespace dist {
static const int SLICE_LENGTH = 0;
static const int SLICE_DATA = 0;
}

static unsigned int id()
{
	int id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	assert(id >= 0);
	return static_cast<unsigned int>(id);
}

static unsigned int nodes()
{
	int nodes;
	MPI_Comm_size(MPI_COMM_WORLD, &nodes);
	assert(nodes >= 0);
	return static_cast<unsigned int>(nodes);
}

static void send_slice(unsigned int destination, util::Slice<uint32_t> slice) {
	auto length = slice.size();
	MPI_Send(&length, 1, MPI_UNSIGNED_LONG, destination, dist::SLICE_LENGTH, MPI_COMM_WORLD);
	MPI_Send(slice.start, slice.size(), MPI_UNSIGNED, destination, dist::SLICE_DATA, MPI_COMM_WORLD);
}

static void recieve_slice(unsigned int source, util::Slice<uint32_t>& slice) {
	MPI_Recv(slice.start, slice.size(), MPI_UNSIGNED_LONG, source, dist::SLICE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

namespace dist
{

void init(int argc, char **argv)
{
	// check MPI_UNSIGNED is a uin32_t and MPI_LONG_UNSIGNED is an uint64_t
	assert(sizeof(unsigned int) == sizeof(uint32_t)); 
	assert(sizeof(unsigned long int) == sizeof(uint64_t)); 

	MPI_Init(&argc, &argv);
}

bool main_process() {
	return id() == 0;
}

Task to_tasks(const vector<uint32_t> &data) {
	Task task;
	auto n_tasks = nodes() - 1; //master node does no work
	auto offsets = cpu::offsets(data, n_tasks);
	task.data = cpu::place_elements(data, offsets, n_tasks);
	
	for (unsigned int i = 0; i < n_tasks; i++) {
		auto length = offsets[i + 1] - offsets[i];
		util::Slice slice(task.data, offsets[i], length);
		task.slices.push_back(slice);
	}

	return task;
}

void fan_out(const Task& tasks) {
	for (unsigned int i=0; i<tasks.slices.size(); i++) {
		auto slice = tasks.slices[i];
		dbg(i);
		send_slice(i+1, slice);
	}
}

void wait_till_done() {
	dbg("");
	MPI_Barrier(MPI_COMM_WORLD);
}

void signal_done() {
	dbg("");
	MPI_Barrier(MPI_COMM_WORLD);
}

void fan_in(Task& tasks) {
	for (unsigned int i=0; i<tasks.slices.size(); i++) {
		auto& slice = tasks.slices[i];
		recieve_slice(i+1, slice);
	}
}

vector<uint32_t> recieve() {
	constexpr int SOURCE = 0;
	uint64_t slice_length;
	MPI_Recv(&slice_length, 1, MPI_UNSIGNED_LONG, SOURCE, dist::SLICE_LENGTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	vector<uint32_t> data;
	data.resize(slice_length);

	MPI_Recv(data.data(), slice_length, MPI_UNSIGNED, SOURCE, dist::SLICE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	return data;
}

void send(const vector<uint32_t>& data) {
	constexpr int DESTINATION = 0;
	MPI_Send(data.data(), data.size(), MPI_UNSIGNED, DESTINATION, dist::SLICE_DATA, MPI_COMM_WORLD);
}

void cleanup() {
    MPI_Finalize();
}

} // namespace dist
