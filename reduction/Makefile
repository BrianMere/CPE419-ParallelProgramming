include ../Makefile
CC = nvcc 
CFLAGS = -Xcompiler -fopenmp -g -lc -lm

build/%.o: src/%.cu $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@