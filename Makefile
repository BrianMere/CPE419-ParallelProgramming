LAB		= $(notdir $(CURDIR)).bin
CC 		= gcc
CFLAGS 	= -lpthread -fpic -std=gnu99

OBJECTS = $(addprefix build/, $(addsuffix .o, $(basename $(notdir $(wildcard src/*.c)))))
HEADERS = $(wildcard src/*.h)

all: build $(LAB)

build:
	mkdir -p build

$(LAB): $(OBJECTS)
	$(CC) -o $(LAB) $(OBJECTS) $(CFLAGS)

build/%.o: src/%.c $(HEADERS)
	$(CC) -c $< $(CFLAGS) -o $@

clean:
	rm -rf build $(LAB)
