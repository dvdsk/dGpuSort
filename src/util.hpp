#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "dbg.h" // compile without warnings enabled
#include <cassert>
#include <cstdint>
#include <vector>

namespace util
{

template <typename T> class SliceIterator {
	template <typename U> friend class Slice;

    public:
	CUDA_CALLABLE SliceIterator(T *_pos, uint64_t _len)
		: pos(_pos), len(_len){};
	CUDA_CALLABLE SliceIterator &operator++()
	{
		pos += 1;
		return *this;
	}
	CUDA_CALLABLE T &operator*()
	{
		return const_cast<T &>(*pos);
	}
	CUDA_CALLABLE bool operator!=(SliceIterator &it)
	{
		return pos != it.pos;
	}

    private:
	T *pos;
	uint64_t len;
};

template <typename T> class Slice {
    public:
	// start of slice, pointer
	T *start;
	// length of slice in elements
	uint64_t len;
	CUDA_CALLABLE Slice(T *_start, uint64_t _len)
	{
		start = _start;
		len = _len;
	}
	Slice(std::vector<T> &vec, std::size_t offset, std::size_t _len)
	{
		assert(offset + _len < vec.size());
		start = vec.data() + offset;
		len = _len;
	}
	CUDA_CALLABLE T &operator[](std::size_t idx)
	{
		assert(idx < len);
		return const_cast<T &>(*(start + idx));
	}
	CUDA_CALLABLE SliceIterator<T> begin()
	{
		return SliceIterator(start, len);
	}
	CUDA_CALLABLE SliceIterator<T> end()
	{
		return SliceIterator(start + len - 1, len);
	}
};

template <typename T> std::vector<T> filled_vec(uint64_t len);
std::vector<uint32_t> random_array(long unsigned int n, long unsigned int seed);
bool is_sorted(std::vector<uint32_t> &data);

} // namespace util

template class util::Slice<uint32_t>;
template class util::Slice<uint64_t>;
