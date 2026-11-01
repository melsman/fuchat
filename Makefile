FUTHARK ?= futhark
SMLFUT ?= smlfut
MLKIT ?= mlkit
CC ?= cc
AR ?= ar
CFLAGS ?= -O2

.PHONY: check-sml clean-sml

build:
	mkdir -p build

build/qwen.c build/qwen.h build/qwen.json: qwen-f32.fut | build
	$(FUTHARK) c --library $< -o build/qwen

build/qwen.sig build/qwen.sml build/qwen.smlfut.c: build/qwen.json
	$(SMLFUT) --target=mlkit $< -o build

build/qwen.o: build/qwen.c build/qwen.h
	$(CC) $(CFLAGS) -c $< -o $@

build/qwen.smlfut.o: build/qwen.smlfut.c build/qwen.h
	$(CC) $(CFLAGS) -c $< -o $@

build/libqwen.a: build/qwen.o build/qwen.smlfut.o
	$(AR) rcs $@ $^

build/check: sml/check.mlb sml/check.sml build/qwen.sig build/qwen.sml build/libqwen.a
	$(MLKIT) -o $@ -libdirs $(CURDIR)/build -libs 'qwen,m,c' sml/check.mlb

check-sml: build/check
	./build/check

clean-sml:
	rm -rf build sml/MLB
