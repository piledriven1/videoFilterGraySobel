# Compiler
CC := g++
# Compiler flags
CFLAGS := -O3 -Wall -Werror -pedantic -lpthread -march=armv8-a -g -isystem /usr/include/opencv4
# Linker flags
LDFLAGS := $(shell pkg-config --libs opencv4)

# Name of the compiled program
MAIN := filter
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
# Binary output name
OUTPUT := $(MAIN)
# Headers
HEAD := $(wildcard *.hpp)

.PHONY: all clean

all : $(OUTPUT)

# Executable file handling
$(OUTPUT): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(MAIN) $(OBJECTS) $(LDFLAGS)

# Binary file handling
%.o: %.cpp $(HEAD)
	$(CC) $(CFLAGS) -c $< -o $@

# Deletes all previously compiled executables and binaries
clean:
	rm -f *.o $(MAIN)
