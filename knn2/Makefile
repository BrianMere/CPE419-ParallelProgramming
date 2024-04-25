include ../Makefile
CC = g++
CFLAGS +=  -fopenmp -std=c++17 -g
# CFLAGS +=  -fcf-protection=none
build/%.o: src/%.cpp $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@