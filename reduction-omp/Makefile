include ../Makefile
CC = g++
CFLAGS = -fopenmp -g -lc -lm

build/%.o: src/%.cpp $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@