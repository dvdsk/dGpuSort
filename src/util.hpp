#pragma once
#include "dbg.h" // compile without warnings enabled
#include <cassert>
#include <cstdint>
#include <vector>

namespace util
{

class Slice {
	class iterator {
		friend class Slice;

	    public:
		iterator(uint32_t *_pos, uint64_t _len) : pos(_pos), len(_len){};
		iterator &operator++()
		{
			pos += 1;
			return *this;
		}
		uint32_t &operator*()
		{
			return const_cast<uint32_t&>(*pos);
		}
		bool operator!=(iterator &it)
		{
			return pos != it.pos;
		}

	    private:
		uint32_t *pos;
		uint64_t len;
	};

    public:
	// start of slice, pointer
	uint32_t *start;
	// length of slice in elements
	uint64_t len;
	iterator begin()
	{
		return iterator(start, len);
	}
	iterator end()
	{
		return iterator(start + len - 1, len);
	}
};

std::vector<uint32_t> filled_vec(uint64_t len);
std::vector<uint32_t> random_array(long unsigned int n, long unsigned int seed);
bool is_sorted(std::vector<uint32_t> &data);

} // namespace util
