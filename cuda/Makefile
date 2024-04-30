include ../Makefile
CC = nvcc
CFLAGS = 

build/%.o: src/%.cu $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@