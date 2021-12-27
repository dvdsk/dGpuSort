#include <cassert>
#include <cstdint>
#include <limits>
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

namespace dist
{
static const int SLICE_LENGTH = 0;
static const int SLICE_DATA = 0;
} // namespace dist

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

// mpi has a maximum send size of u32::max elements, this bypasses that
// by sending in batches, the size must already be known on the other size
static void mpi_send(unsigned int destination, uint32_t *start, size_t size)
{
	const auto max = std::numeric_limits<uint32_t>::max();
	unsigned int send = 0;
	while (size - send > max) {
		dbg("NOPE");
		auto offset = start + send;
		MPI_Send(offset, size, MPI_UNSIGNED, destination,
			 dist::SLICE_DATA, MPI_COMM_WORLD);
		send += max;
	}
	auto offset = start + send;
	MPI_Send(offset, size, MPI_UNSIGNED, destination, dist::SLICE_DATA,
		 MPI_COMM_WORLD);
}

// size has te be known in advance, memory needs to be preallocated
static void mpi_recv(unsigned int source, uint32_t *start, size_t size)
{
	dbg(source, size);
	const auto max = std::numeric_limits<uint32_t>::max();
	unsigned int got = 0;
	while (size - got > max) {
		dbg("NOPE");
		auto offset = start + got;
		auto ok = MPI_Recv(offset, size, MPI_UNSIGNED, source,
				   dist::SLICE_DATA, MPI_COMM_WORLD,
				   MPI_STATUS_IGNORE);
		assert(ok == MPI_SUCCESS);
		got += max;
	}
	auto offset = start + got;
	auto ok = MPI_Recv(offset, size, MPI_UNSIGNED, source, dist::SLICE_DATA,
			   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	assert(ok == MPI_SUCCESS);
}

static void send_slice(unsigned int destination, util::Slice<uint32_t> slice)
{
	dbg("");
	auto size = slice.size();
	MPI_Send(&size, 1, MPI_UNSIGNED_LONG, destination, dist::SLICE_LENGTH,
		 MPI_COMM_WORLD);
	mpi_send(destination, const_cast<uint32_t *>(slice.start),
		 slice.size());
}

static void recieve_slice(unsigned int source, util::Slice<uint32_t> &slice)
{
	dbg("");
	auto buffer = const_cast<uint32_t *>(slice.start);
	mpi_recv(source, buffer, slice.size());
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

bool main_process()
{
	return id() == 0;
}

Task to_tasks(const vector<uint32_t> &data)
{
	Task task;
	auto n_tasks = nodes();
	auto offsets = cpu::offsets(data, n_tasks);
	task.data = cpu::place_elements(data, offsets, n_tasks);
	for (unsigned int i = 0; i < n_tasks; i++) {
		auto length = offsets[i + 1] - offsets[i];
		util::Slice slice(task.data, offsets[i], length);
		task.slices.push_back(slice);
	}

	return task;
}

void fan_out(Task &tasks)
{
	for (unsigned int i = 1; i < tasks.slices.size(); i++) {
		auto slice = tasks.slices[i];
		send_slice(i, slice);
	}
}

void wait_till_done()
{
	MPI_Barrier(MPI_COMM_WORLD);
}

void signal_done()
{
	MPI_Barrier(MPI_COMM_WORLD);
}

void fan_in(Task &tasks)
{
	for (unsigned int i = 1; i < tasks.slices.size(); i++) {
		auto &slice = tasks.slices[i];
		recieve_slice(i, slice);
		dbg(i);
		util::assert_sort(slice, slice);
		dbg("post");
	}
}

vector<uint32_t> recieve()
{
	constexpr int SOURCE = 0;
	uint64_t slice_length;
	auto ok =
		MPI_Recv(&slice_length, 1, MPI_UNSIGNED_LONG, SOURCE,
			 dist::SLICE_LENGTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	assert(ok == MPI_SUCCESS);

	vector<uint32_t> data;
	data.resize(slice_length);
	mpi_recv(SOURCE, data.data(), slice_length);

	return data;
}

void send(const vector<uint32_t> &data)
{
	dbg("");
	constexpr int DESTINATION = 0;
	dbg(data);
	util::assert_sort(data, data);
	mpi_send(DESTINATION, const_cast<uint32_t *>(data.data()), data.size());
}

void cleanup()
{
	MPI_Finalize();
}

} // namespace dist
