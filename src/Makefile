export CC = gcc
export CXX = g++
export CFLAGS = -w -O3 -pthread -std=c++0x

LDFLAGS = -lm -lgsl -lgslcblas -larmadillo -fpermissive
INSTALL_PATH=bin/
BIN = onlinectl
OBJ = onlinectl-estimate.o onlinectl-inference.o onlinectl-learn.o utils.o gsl-wrappers.o params.o
.PHONY: clean all

all: $(BIN)

onlinectl:onlinectl.cc onlinectl-estimate.o onlinectl-inference.o onlinectl-learn.o utils.o onlinectl.h gsl-wrappers.o params.o
onlinectl-estimate.o:onlinectl-estimate.cc utils.h onlinectl.h onlinectl-estimate.h
onlinectl-inference.o: onlinectl-inference.cc utils.h onlinectl-inference.h onlinectl.h
onlinectl-learn.o: onlinectl-learn.cc utils.h onlinectl-learn.h onlinectl.h
utils.o: utils.cc
gsl-wrappers.o: gsl-wrappers.cc
params.o: params.cc

$(BIN) : 
	$(CXX) $(filter %.cpp %.o %.cc, $^) $(CFLAGS) -o $@ $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(filter %.cpp %.cc, $^) $(LDFLAGS)

install:
	cp -f -r $(BIN) $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
