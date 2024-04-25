include ../Makefile
CC = g++
CFLAGS +=  -fopenmp
# CFLAGS +=  -fcf-protection=none
build/%.o: src/%.cpp $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@