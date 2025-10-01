# Compiler
CC := g++
# Compiler flags
CFLAGS := -Wall -Werror -pedantic -g $(shell pkg-config --cflags opencv4)
# Linker flags
LDFLAGS := $(shell pkg-config --libs opencv4)

# Name of the compiled program
MAIN := extFrame
# Binary output name
OUTPUT := $(MAIN)
# Headers
HEAD := $(wildcard *.h)

.PHONY: all, grey, sobel, clean

all : $(OUTPUT)

# Executable file handling
$(OUTPUT): $(MAIN).o
	$(CC) $(CFLAGS) -o $(MAIN) $(wildcard *.o) $(LDFLAGS)

# Binary file handling
$(MAIN).o: $(wildcard *.c) $(HEAD)
	$(CC) $(CFLAGS) -c $(MAIN).cpp

# Deletes all previously compiled executables and binaries
clean:
	rm *.jpg *.o $(MAIN)
