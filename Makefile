# Compiler
CC := g++
# Compiler flags
CFLAGS := -O0 -Wall -Werror -pedantic -lpthread -march=armv8-a -g $(shell pkg-config --cflags opencv4)
# Linker flags
LDFLAGS := $(shell pkg-config --libs opencv4)

# Name of the compiled program
MAIN := extFrame
SOURCES := $(wildcard *.cpp) $(wildcard *.c)
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
	rm *.o $(MAIN)
