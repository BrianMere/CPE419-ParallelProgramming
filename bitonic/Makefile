include ../Makefile
CC = nvcc 
CFLAGS = -Xcompiler -fopenmp -G -lc -lm

build/%.o: src/%.cu $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@