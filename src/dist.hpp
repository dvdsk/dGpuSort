#pragma once

#include <cstdint>
#include <vector>

namespace dist
{

struct Task {
	bool use_gpu;
	std::vector<uint32_t> data;
};

void init(int argc, char **argv);
bool main_process();
std::vector<uint32_t> into_buckets(std::vector<uint32_t> data);
void fan_out(std::vector<uint32_t> data);
void wait_till_done();
std::vector<uint32_t> fan_in();
Task recieve();
void send(std::vector<uint32_t>);
void cleanup();

} // namespace dist
