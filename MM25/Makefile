MAIN = main
OBJS = main.o MM25.o

CC = mpicc
CFLAGS = -O3 -xCORE-AVX2 -qopenmp -std=c99
MKLLIB = -mkl=parallel

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) -o $@ $^ $(MKLLIB)

.SUFFIXES: .c .o
.c.o:
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm $(MAIN) *.o

localtest: localtest.o MM25.o
	$(CC) -o $@ $^ $(MKLLIB);./localtest;rm localtest
