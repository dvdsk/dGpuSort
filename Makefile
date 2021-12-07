# $@ is a macro that refers to the target
# $< is a macro that refers to the first dependency
# $^ is a macro that refers to all dependencies
# % is a macro to make a pattern that we want to watch in both the target and the dependency

RUNTIME_ENV := ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0

# -----------------------------------------------------------------------------
#  Setting up data and dirs
# -----------------------------------------------------------------------------

directories:
	mkdir -p build
	mkdir -p target
	mkdir -p data

# -----------------------------------------------------------------------------
# Building C++ program
# -----------------------------------------------------------------------------
include compile.mk

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

release/%.mtx: target/release
	target/release $<

debug/%.mtx: target/debug
	target/debug $<

test: target/test_gpu target/test_cpu
	$(RUNTIME_ENV) target/test_gpu
	# $(RUNTIME_ENV) target/test_cpu
	
test_gpu: target/test_gpu
	prun -np 1 -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/test_gpu

# -----------------------------------------------------------------------------
# Util
# -----------------------------------------------------------------------------

.PHONY: all clean directories run r
all: parallel

clean:
	rm -f build/*.o 
	rm -f target/*

clean_all: clean
	rm -rf data/*


r: run #ease of use alias
run: sequential
	./sequential "data/nopoly.mtx"
