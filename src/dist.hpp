#pragma once

#include "util.hpp"
#include <cstdint>
#include <vector>
#include <mpi.h>

namespace dist
{

struct SendReq {
	MPI_Request length;
	MPI_Request data;
};

struct Task {
	std::vector<uint32_t> data;
	std::vector<util::Slice<uint32_t>> slices;
	std::vector<SendReq> send_req;
};

void init(int argc, char **argv);
bool main_process();
Task to_tasks(const std::vector<uint32_t> &data);
void fan_out(Task& tasks);
void wait_till_done();
void signal_done();
/* after the fan in the data field of Tasks contains sorted data*/
void fan_in(Task& tasks);
std::vector<uint32_t> recieve();
void send(const std::vector<uint32_t>& data);
void cleanup();

} // namespace dist
