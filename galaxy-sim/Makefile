include ../Makefile
CC = nvcc 
CFLAGS = --default-stream legacy -Xcompiler -fopenmp -Xcompiler -g -D_DEFAULT_SOURCE -lc

build/%.o: src/%.cu $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@