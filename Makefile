# $@ is a macro that refers to the target
# $< is a macro that refers to the first dependency
# $^ is a macro that refers to all dependencies
# % is a macro to make a pattern that we want to watch in both the target and the dependency

RUNTIME_ENV := ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0

all: sort
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

sort: target/release
	mv target/release sort

test: target/release
	prun -np 1 -script $(PRUN_ETC)/prun-openmpi -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/release 5 80000000 1
	# prun -np 2 -script $(PRUN_ETC)/prun-openmpi -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/release 5 80000000 0
	# prun -np 4 -script $(PRUN_ETC)/prun-openmpi -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/release 5 80000000 0
	# prun -np 8 -script $(PRUN_ETC)/prun-openmpi -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/release 5 80000000 0

test2: target/debug
	prun -np 15 -script $(PRUN_ETC)/prun-openmpi -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/debug 5 1600000 0
	
test_gpu: target/test_gpu
	prun -np 1 -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/test_gpu

dist_cpu/%: target/test_dist
	prun -np % -script $(PRUN_ETC)/prun-openmpi -native '-C TitanX --gres=gpu:1' $(RUNTIME_ENV) `pwd`/target/test_dist

# -----------------------------------------------------------------------------
# Util
# -----------------------------------------------------------------------------

.PHONY: clean directories run r

clean:
	rm -f build/*.o 
	rm -f target/*

clean_all: clean
	rm -rf data/*
