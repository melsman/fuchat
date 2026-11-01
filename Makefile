FUTHARK ?= futhark
FUTHARK_BACKEND ?= c
TYPE ?= f32
SMLFUT ?= smlfut
MLKIT ?= mlkit
SMLPKG ?= smlpkg
PYTHON ?= python3
CC ?= cc
AR ?= ar
CFLAGS ?= -O2
CPPFLAGS ?=
CUDA_HOME ?= /usr/local/cuda
MODEL ?= Qwen3-0.6B-f16.gguf
BUILD = build/$(FUTHARK_BACKEND)-$(TYPE)
UNICODE_MLB = lib/github.com/diku-dk/sml-unicode/unicode-normalise.mlb
SML_SOURCES = $(wildcard sml/*.sml sml/*.sig) sml/precision/$(TYPE).sml
LIBDIRS = $(CURDIR)/$(BUILD)
LIBS = qwen,host,m,c,dl

ifeq ($(filter $(TYPE),f32 f16),)
$(error TYPE must be f32 or f16)
endif
ifeq ($(FUTHARK_BACKEND),cuda)
CPPFLAGS += -I$(CUDA_HOME)/include
LIBDIRS := $(LIBDIRS),$(CUDA_HOME)/lib64
LIBS := $(LIBS),cuda,cudart,nvrtc
else ifeq ($(FUTHARK_BACKEND),multicore)
CFLAGS += -pthread
LIBS := $(LIBS),pthread
else ifneq ($(FUTHARK_BACKEND),c)
$(error Supported FUTHARK_BACKEND values are c, multicore, and cuda)
endif

.PHONY: all build/chat-sml check-sml check-parity check-model check-conversation clean-sml deps check-host
all: build/chat-sml

# Compatibility shortcut; explicit variant paths remain available side by side.
build/chat-sml: $(BUILD)/chat-sml
	ln -sfn $(FUTHARK_BACKEND)-$(TYPE)/chat-sml $@

build $(BUILD):
	mkdir -p $@

deps: build/deps.stamp $(UNICODE_MLB)

build/deps.stamp: sml.pkg | build
	$(SMLPKG) sync
	touch $@

$(UNICODE_MLB): | build/deps.stamp
	@test -f $@ || $(SMLPKG) sync

$(BUILD)/qwen.c: qwen-$(TYPE).fut Makefile | $(BUILD)
	$(FUTHARK) $(FUTHARK_BACKEND) --library $< -o $(BUILD)/qwen

$(BUILD)/qwen.h $(BUILD)/qwen.json: $(BUILD)/qwen.c

$(BUILD)/qwen.sml: $(BUILD)/qwen.json
	$(SMLFUT) --target=mlkit $< -o $(BUILD)

$(BUILD)/qwen.sig $(BUILD)/qwen.smlfut.c: $(BUILD)/qwen.sml

$(BUILD)/qwen.o: $(BUILD)/qwen.c $(BUILD)/qwen.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/qwen.smlfut.o: $(BUILD)/qwen.smlfut.c $(BUILD)/qwen.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/large-file.o: sml/large-file.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/libhost.a: $(BUILD)/large-file.o
	$(AR) rcs $@ $^

$(BUILD)/libqwen.a: $(BUILD)/qwen.o $(BUILD)/qwen.smlfut.o
	$(AR) rcs $@ $^

build/generate-tools: tools/generate-tools.mlb tools/GenerateTools.sml sml/Json.sml sml/Unicode.sml sml/UnicodeCategories.sml sml/Util.sml | build
	$(MLKIT) -o $@ tools/generate-tools.mlb

build/generate-project: tools/generate-project.mlb tools/GenerateProject.sml | build
	$(MLKIT) -o $@ tools/generate-project.mlb

$(BUILD)/ToolEntries.sml: $(BUILD)/qwen.json build/generate-tools
	./build/generate-tools $< > $@.tmp
	mv $@.tmp $@

$(BUILD)/chat.mlb: sml/chat.mlb $(SML_SOURCES) build/generate-project | $(BUILD)
	./build/generate-project "$(CURDIR)" "$(BUILD)" "$(TYPE)" $< > $@.tmp
	mv $@.tmp $@

$(BUILD)/session.mlb: tests/session.mlb tests/session-main.sml $(SML_SOURCES) build/generate-project | $(BUILD)
	./build/generate-project "$(CURDIR)" "$(BUILD)" "$(TYPE)" $< > $@.tmp
	mv $@.tmp $@

$(BUILD)/host.mlb: tests/host.mlb tests/host-main.sml $(SML_SOURCES) build/generate-project | $(BUILD)
	./build/generate-project "$(CURDIR)" "$(BUILD)" "$(TYPE)" $< > $@.tmp
	mv $@.tmp $@

$(BUILD)/host-test: $(BUILD)/host.mlb tests/host-main.sml $(SML_SOURCES) $(BUILD)/qwen.sig $(BUILD)/qwen.sml $(BUILD)/libqwen.a $(BUILD)/libhost.a
	$(MLKIT) -o $@ -libdirs "$(LIBDIRS)" -libs '$(LIBS)' $(BUILD)/host.mlb

check-host: $(BUILD)/chat-sml $(BUILD)/host-test
	$(PYTHON) tests/host.py --exe ./$(BUILD)/chat-sml --host-exe ./$(BUILD)/host-test --type $(TYPE)

$(BUILD)/check.mlb: sml/check.mlb $(SML_SOURCES) build/generate-project | $(BUILD)
	./build/generate-project "$(CURDIR)" "$(BUILD)" "$(TYPE)" $< > $@.tmp
	mv $@.tmp $@

$(BUILD)/check: $(BUILD)/check.mlb sml/check.sml $(SML_SOURCES) $(BUILD)/qwen.sig $(BUILD)/qwen.sml $(BUILD)/libqwen.a $(BUILD)/libhost.a
	$(MLKIT) -o $@ -libdirs "$(LIBDIRS)" -libs '$(LIBS)' $(BUILD)/check.mlb

$(BUILD)/chat-sml: $(BUILD)/chat.mlb $(BUILD)/ToolEntries.sml $(SML_SOURCES) build/deps.stamp $(UNICODE_MLB) $(BUILD)/qwen.sig $(BUILD)/qwen.sml $(BUILD)/libqwen.a $(BUILD)/libhost.a
	$(MLKIT) -o $@ -libdirs "$(LIBDIRS)" -libs '$(LIBS)' $(BUILD)/chat.mlb

$(BUILD)/session-test: $(BUILD)/session.mlb tests/session-main.sml $(SML_SOURCES) build/deps.stamp $(UNICODE_MLB) $(BUILD)/ToolEntries.sml $(BUILD)/qwen.sig $(BUILD)/qwen.sml $(BUILD)/libqwen.a $(BUILD)/libhost.a
	$(MLKIT) -o $@ -libdirs "$(LIBDIRS)" -libs '$(LIBS)' $(BUILD)/session.mlb

check-sml: $(BUILD)/check
	./$(BUILD)/check

check-conversation: $(BUILD)/chat-sml $(BUILD)/session-test
	$(PYTHON) tests/conversation.py --exe ./$(BUILD)/chat-sml --session-exe ./$(BUILD)/session-test

check-parity: $(BUILD)/chat-sml $(BUILD)/session-test check-host
	$(PYTHON) tests/parity.py --exe ./$(BUILD)/chat-sml --backend $(FUTHARK_BACKEND) --type $(TYPE)
	$(PYTHON) tests/conversation.py --exe ./$(BUILD)/chat-sml --session-exe ./$(BUILD)/session-test

check-model: $(BUILD)/chat-sml $(BUILD)/session-test check-host
	$(PYTHON) tests/parity.py --exe ./$(BUILD)/chat-sml --backend $(FUTHARK_BACKEND) --type $(TYPE) --model "$(MODEL)"
	$(PYTHON) tests/conversation.py --exe ./$(BUILD)/chat-sml --session-exe ./$(BUILD)/session-test --model "$(MODEL)"

clean-sml:
	rm -rf build sml/MLB sml/precision/MLB tools/MLB tests/MLB
