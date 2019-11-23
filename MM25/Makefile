MAIN = main
OBJS = main.o MM25.o

CC = mpicc
CFLAGS = -O3 -xCORE-AVX2 -qopenmp -mkl=parallel

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) -o $@ $^

.SUFFIXES: .c .o
.c.o:
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm $(MAIN) *.o
