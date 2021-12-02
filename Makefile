# $@ is a macro that refers to the target
# $< is a macro that refers to the first dependency
# $^ is a macro that refers to all dependencies
# % is a macro to make a pattern that we want to watch in both the target and the dependency

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

bench/%.mtx: data/%.mtx target/release
	target/release $<

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
