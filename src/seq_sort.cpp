#include "seq_sort.hpp"
#include "util.hpp"
#include <cstddef>
#include <limits>
#include <vector>

using Slice = util::Slice<uint32_t>;
using std::size_t;

CUDA_CALLABLE static void swap(uint32_t &a, uint32_t &b)
{
	uint32_t temp = a;
	a = b;
	b = temp;
}

CUDA_CALLABLE static void insert_sort(Slice data) {
   for(size_t i = 1; i < data.size(); i++) {
      size_t key = data[i];
      size_t j = i;
      while(j > 0 && data[j-1] > key) {
         data[j] = data[j-1];
         j--;
      }
      data[j] = key;
   }
}

CUDA_CALLABLE static std::size_t partition(util::Slice<uint32_t> A,
					   std::size_t lo, std::size_t hi)
{
	std::size_t pivot = A[hi];
	auto i = lo - 1;

	for (auto j = lo; j < hi; j++) {
		if (A[j] <= pivot) {
			i++;
			swap(A[i], A[j]);
		}
	}
	swap(A[i + 1], A[hi]);
	return i + 1;
}

CUDA_CALLABLE static void quick(util::Slice<uint32_t> A, std::size_t lo,
				std::size_t hi)
{
	if (lo < hi && hi != std::numeric_limits<std::size_t>::max()) {
		if (hi - lo + 1 < 16) {
			util::Slice<uint32_t> slice(A, lo, hi - lo + 1);
			insert_sort(slice);
		} else {
			std::size_t p = partition(A, lo, hi);
			quick(A, lo, p - 1);
			quick(A, p + 1, hi);
		}
	}
}

namespace seq_sort
{
CUDA_CALLABLE void quick_sort(util::Slice<uint32_t> A)
{
	if (A.size() == 0) {
		return;
	}
	quick(A, 0, A.size() - 1);
}
} // namespace seq_sort
