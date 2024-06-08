include ../Makefile

CC = icpx
CFLAGS = -fsycl -qopenmp -lm -fsycl-targets=nvptx64-nvidia-cuda -Wall
# Make sure to call "source /opt/intel/oneapi/setvars.sh" before calling make...

build/%.o: src/%.cpp $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@

