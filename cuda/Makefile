include ../Makefile
CC = nvcc 
CFLAGS = -Xcompiler -fopenmp -Xcompiler -g

build/%.o: src/%.cu $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@