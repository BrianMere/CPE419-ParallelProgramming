include ../Makefile
CC = gcc
CFLAGS = -fopenmp -g -O3 -std=c99 -lc -Wall -DTRACE -D_DEFAULT_SOURCE

build/%.o: src/%.c $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@