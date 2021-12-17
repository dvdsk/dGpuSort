CXX = g++
CC = gcc
NVCC = nvcc

WARNINGS := \
	-pedantic \
	-Wall \
	-Wextra \
	-Wcast-align \
	-Wcast-qual \
	-Wctor-dtor-privacy \
	-Wdisabled-optimization \
	-Wformat=2 \
	-Winit-self \
	-Wlogical-op \
	-Wmissing-declarations \
	-Wmissing-include-dirs \
	-Wnoexcept \
	-Wold-style-cast \
	-Woverloaded-virtual \
	-Wredundant-decls \
	-Wshadow \
	-Wsign-conversion \
	-Wsign-promo \
	-Wstrict-null-sentinel \
	-Wstrict-overflow=5 \
	-Wswitch-default \
	-Wundef \
	-Wno-unused

BASIC := -std=c++17
HEADERS := -I dependencies/dbg -I /usr/lib/cuda/include

# -Og; enable optimisations that do not interfere with debugging
# -ggdb; enable the call stack for better report format
# -fno-omit-frame-pointer; nicer stack traces in error messages
# -fsanitize=address; link (compile and runtime) memory error detector (heap and stack)
#  	note, adress sanatizer needs to be disabled if running under valgrind
# -fsanitize-address-use-after-scope; more agressive adress sanatizer
#  	note, adress sanatizer needs to be disabled if running under valgrind
# -fdiagnostics-color=always; give me pretty colors
# -D _GLIBCXX_DEBUG=1; use the standard lib in debug mode
# -D DBG_MACRO_NO_WARNING; do not output warning the dbg macro is enabled
# -fuse-ld=lld; use lld linker (has better diagnostics) instead of ld
DIAGNOSTICS := -Og \
	-ggdb \
	-fno-omit-frame-pointer \
	-fdiagnostics-color=always \
	-D _GLIBCXX_DEBUG=1 \
	-D _GLIBCXX_DEBUG_PEDANTIC=1 \
	-D DBG_MACRO_NO_WARNING \
	-fsanitize=address \
	-fsanitize-address-use-after-scope \
	# -fuse-ld=lld \
# link to binutils_dev for nicer stack tracing
DIAGNOSTICS += -lbfd -ldl

# TODO add: https://stackoverflow.com/questions/11312719/how-to-compile-mpi-with-gcc
PERFORMANCE := -O3 -flto -D DBG_MACRO_DISABLE -D DBG_MACRO_NO_WARNING
CUDA_BASIC := $(BASIC)
CUDA_DIAG := -D _GLIBCXX_DEBUG -D _GLIBCXX_DEBUG_PEDANTIC
CUDA_CCFLAGS := $(BASIC) $(HEADERS) $(CUDA_DIAG) -x cu -arch=sm_50 -dc -g -G
CUDA_LDFLAGS := -arch=sm_50 -dlink 
CCFLAGS := $(BASIC) $(HEADERS) $(WARNINGS) $(DIAGNOSTICS)
LDFLAGS := -lcudart -L/usr/lib/cuda/lib64

# -----------------------------------------------------------------------------
#  binaries
# -----------------------------------------------------------------------------

OBJ := util.o gpu.o gpu_code.o cpu.o seq_sort.o
DEPS := $(OBJ) backward.o # linking to backward gives stack backtraces
target/debug: $(addprefix build/, main.o $(DEPS))
	$(CXX) $(CCFLAGS) $(LDFLAGS) -o $@ $^ 

target/release: override CCFLAGS := $(BASIC) $(WARNINGS) $(PERFORMANCE)
target/release: override CUDA_DIAG := 
target/release: $(addprefix build/, main.o $(OBJ))
	$(CXX) $(CCFLAGS) $(LDFLAGS) -o $@ $^

target/test_gpu: $(addprefix build/, test_gpu.o $(DEPS))
	$(CXX) $(CCFLAGS) $(LDFLAGS) -o $@ $^

target/test_cpu: $(addprefix build/, test_cpu.o $(DEPS))
	$(CXX) $(CCFLAGS) $(LDFLAGS) -o $@ $^

target/test_seg: $(addprefix build/, test_seg.o $(DEPS))
	$(CXX) $(CCFLAGS) $(LDFLAGS) -o $@ $^

target/test_dist: $(addprefix build/, test_dist.o $(DEPS))
	$(CXX) $(CCFLAGS) $(LDFLAGS) -o $@ $^

target/test_slice: $(addprefix build/, test_slice.o $(DEPS))
	$(CXX) $(CCFLAGS) $(LDFLAGS) -o $@ $^
# -----------------------------------------------------------------------------
#  dependencies
# -----------------------------------------------------------------------------

build/backward.o: override CCFLAGS := -std=c++17
build/backward.o: \
	dependencies/backward/backward.cpp \
	dependencies/backward/backward.hpp
	$(CXX) $(CCFLAGS) -c -o $@ $<

# -----------------------------------------------------------------------------
#  source files
# -----------------------------------------------------------------------------

# generic
build/%.o: src/%.c | directories
	$(CXX) $(CCFLAGS) -c -o $@ $<
build/%.o: src/%.cpp | directories
	$(CXX) $(CCFLAGS) -c -o $@ $<

# gpu code
build/gpu.o: src/gpu.cu | directories
	$(NVCC) $(CUDA_CCFLAGS) --output-file $@ --compile $^
build/gpu_code.o: build/gpu.o | directories
	$(NVCC) $(CUDA_LDFLAGS) $^ --output-file $@

# test dir
build/test_cpu.o: src/tests/cpu.cpp | directories
	$(CXX) $(CCFLAGS) -c -o $@ $^ 
build/test_gpu.o: src/tests/gpu.cpp | directories
	$(CXX) $(CCFLAGS) -c -o $@ $^ 
build/test_seq.o: src/tests/seq.cpp | directories
	$(CXX) $(CCFLAGS) -c -o $@ $^ 
build/test_dist.o: src/tests/dist.cpp | directories
	$(CXX) $(CCFLAGS) -c -o $@ $^ 
build/test_slice.o: src/tests/slice.cpp | directories
	$(CXX) $(CCFLAGS) -c -o $@ $^ 
