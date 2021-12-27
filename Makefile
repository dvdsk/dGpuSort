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

test: target/release
	prun -np 2 -script $(PRUN_ETC)/prun-openmpi -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/release 5 4000000 0
	
test_gpu: target/test_gpu
	prun -np 1 -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/test_gpu

dist_cpu/%: target/test_dist
	prun -np % -script $(PRUN_ETC)/prun-openmpi -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/test_dist

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
