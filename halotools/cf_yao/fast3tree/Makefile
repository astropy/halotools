CC = gcc
CFLAGS=-m64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_BSD_SOURCE -D_POSIX_SOURCE -D_POSIX_C_SOURCE=200809L -D_SVID_SOURCE -D_DARWIN_C_SOURCE -Wall -fno-math-errno -fPIC
LDFLAGS=-shared
OFLAGS=-lm -O3 -std=c99

OBJS = fast3tree_3d.so

all: $(OBJS)

%.so: %.c fast3tree.c
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@ $(OFLAGS)

clean:
	rm -f $(OBJS)

.PHONY: clean
