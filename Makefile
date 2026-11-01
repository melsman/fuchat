FUTHARK ?= futhark
FUTHARK_BACKEND ?= c
SMLFUT ?= smlfut
MLKIT ?= mlkit
SMLPKG ?= smlpkg
PYTHON ?= python3
CC ?= cc
AR ?= ar
CFLAGS ?= -O2
MODEL ?= Qwen3-0.6B-f16.gguf
UNICODE_MLB = lib/github.com/diku-dk/sml-unicode/unicode-normalise.mlb
SML_SOURCES = $(wildcard sml/*.sml)

.PHONY: all check-sml check-parity check-model clean-sml deps
all: build/chat-sml

build:
	mkdir -p build

deps: $(UNICODE_MLB)

$(UNICODE_MLB): sml.pkg
	$(SMLPKG) sync

# Each generator has one primary target, so parallel make invokes it once.
build/qwen.c: qwen-f32.fut Makefile | build
	$(FUTHARK) $(FUTHARK_BACKEND) --library $< -o build/qwen

build/qwen.h build/qwen.json: build/qwen.c

build/qwen.sml: build/qwen.json
	$(SMLFUT) --target=mlkit $< -o build

build/qwen.sig build/qwen.smlfut.c: build/qwen.sml

build/qwen.o: build/qwen.c build/qwen.h
	$(CC) $(CFLAGS) -c $< -o $@

build/qwen.smlfut.o: build/qwen.smlfut.c build/qwen.h
	$(CC) $(CFLAGS) -c $< -o $@

build/large-file.o: sml/large-file.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/libhost.a: build/large-file.o
	$(AR) rcs $@ $^

build/libqwen.a: build/qwen.o build/qwen.smlfut.o
	$(AR) rcs $@ $^

build/check: sml/check.mlb sml/check.sml build/qwen.sig build/qwen.sml build/libqwen.a
	$(MLKIT) -o $@ -libdirs $(CURDIR)/build -libs 'qwen,m,c' sml/check.mlb

build/chat-sml: sml/chat.mlb $(SML_SOURCES) $(UNICODE_MLB) build/qwen.sig build/qwen.sml build/libqwen.a build/libhost.a
	$(MLKIT) -o $@ -libdirs $(CURDIR)/build -libs 'qwen,host,m,c' sml/chat.mlb

check-sml: build/check
	./build/check

# Tiny, deterministic GGUF fixture; no model download needed. Python is used
# only as the independent reference, never by the SML application.
check-parity: build/chat-sml
	$(PYTHON) tests/parity.py --exe ./build/chat-sml

check-model: build/chat-sml
	$(PYTHON) tests/parity.py --exe ./build/chat-sml --model $(MODEL)

clean-sml:
	rm -rf build sml/MLB
