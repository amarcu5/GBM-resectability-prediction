csrc = $(wildcard src/*.c)
ccsrc = $(wildcard src/*.cc)
obj = $(csrc:.c=.o) $(ccsrc:.cc=.o)

testsrc = $(wildcard src/tests/*.cc)
testobj = $(testsrc:.cc=.o) $(filter-out src/main.o, $(obj))

LDFLAGS = -lfann -lpthread
CXXFLAGS = -O3 -std=c++14 -Wall -DMULTITHREAD

.PHONY: build build-run build-test build-doc run test clean

build: build-run build-test build-doc

build-run: ./bin/run

build-test: ./bin/test

build-doc: ./docs/_build

./bin/run: $(obj)
	$(CXX) -o ./bin/run $^ $(LDFLAGS)

./bin/test: $(testobj)
	$(CXX) -o ./bin/test $^ $(LDFLAGS)
	
./docs/_build: $(wildcard src/*.h)
	cd ./docs/ && $(MAKE) html

run: build-run
	./bin/run ./data/raw/*.dat
	
test: build-test
	./bin/test

clean:
	rm -f $(obj) $(testobj) ./bin/run ./bin/test
	cd ./docs/ && $(MAKE) clean
