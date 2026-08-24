CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic
INCLUDES = -I../components/latentmesh_air/include
PURE_SOURCES = \
	../components/latentmesh_air/src/lm_air_ble_frag.c \
	../components/latentmesh_air/src/lm_air_crc.c \
	../components/latentmesh_air/src/lm_air_kiss.c \
	../components/latentmesh_air/src/lm_air_policy.c
PORTABLE_SOURCES = \
	../../../c/src/air.c \
	../../../c/src/fec.c \
	../../../c/src/frame.c \
	../../../c/src/modem.c \
	../../../c/src/neural.c

.PHONY: all test clean
all: test

test: test_pure test_portable_contract
	./test_pure
	./test_portable_contract
	$(RM) test_pure test_portable_contract

test_pure: test_pure.c $(PURE_SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@

test_portable_contract: test_portable_contract.c $(PORTABLE_SOURCES)
	$(CC) $(CFLAGS) -I../../../c/include $^ -lm -o $@

clean:
	rm -f test_pure test_portable_contract
