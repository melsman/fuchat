# fuchat

## Introduction

Qwen3 LLM inference using the Futhark language + demo chat application

The inference engine has support for KV Cache & Prompt extension.

The chat app has support for
 - user/assistant chat
 - Thinking mode on/off
 - Tool calling of simple futhark entry points

The default model is Qwen3-0.6B. Memory use depends on precision and context
size; see the Standard ML host instructions below for a 4GB GPU configuration.


## Usage

You will need
 * a recent nightly release of [Futhark](https://futhark-lang.org/)
 * a python venv with the requirements installed
 
As a first step, compile 
* `futhark {backend} --server qwen-f32.fut -o qwen-f32` for the float32 version (default)
* `futhark {backend} --server qwen-f16.fut -o qwen-f16` for the float16 version
You should probably compile both version so everything's ready for chat tests.

Then you can start the chat with `python chat.py --tools` (or `python chat.py --help` if you want to see the parameters)

and hopefully, after downloading the model weights ( ~1.5GB in float16 format), start chatting with `fuchat` :-)

## Performance

It was developed and tested on an AMD 6700XT with 12GB of VRAM, with the Futhark `hip` backend.

tests are done without tools defined and with thinking disabled 

The 'hedgehog' prompt is "write a story about a hedgehog". The statistics are given for the second run.

| gguf  | command | prompt t/s | generation t/s |
| - | - | - | - |
| Qwen3-0.6B-f16.gguf  | python .\chat.py --type f32 --cnt 50 | 331.9 | 23.7 |
| Qwen3-0.6B-f32.gguf  | llama-cli --model Qwen3-0.6B-f32.gguf -ctv f32 -ctk f32 | 2189.5 | 56.5 |
| Qwen3-0.6B-f16.gguf  | python .\chat.py --type f16 --cnt 50  | 348.9 | 29.0 |
| Qwen3-0.6B-f16.gguf  | llama-cli --model Qwen3-0.6B-f16.gguf -ctv f16 -ctk f16 | 5107.3 | 145.9 |

Note1: the Qwen3-0.6B-f32.gguf model was prepared from Qwen3-0.6B-f16.gguf with the command `llama-quantize Qwen3-0.6B-f16.gguf f32`
Note2: llama-cli is launched with these additional parameters `--temperature 0 --top-k 1 --reasoning off`. The llama.cpp kvcache quantization is adapted with `-ctv` and `-ctk` to match what is happening in fuchat

Observations:
 - prompt t/s during the first run are lower by a large factor both on fuchat & llama.cpp. It could be due to the first calls of the kernels & some cache on gpu
 - it would be interesting to check how the prompt t/s of llama.cpp is calculated vs fuchat to understand the huge difference

A pure f32 version before implementing KV Cache was running at 2-5 tokens/sec so the Futhark "update in-place" mechanism brings a great performance boost for this type of caching.

It is impressive that Futhark can reach ~ half the speed of llama.cpp on f32 with a one file, typed checked, standalone ~ 100 lines .fut file (without comments) !

If you are interested on improving the speed of `fuchat`, a blog post was written on [how to benchmark fuchat](https://futhark-lang.org/blog/2026-05-22-benchmarking-a-real-futhark-application.html) using futhark's tools.


## Acknowledgements

This is largely inspired by [llaf](https://github.com/BobMcDear/) that created a lightweight GPT2 implementation in Futhark

It wouldn't exist without the help and support of [Troels Henriksen] on subtleties around uniqueness and in-place updates.

## Citations

```bibtex
@software{The_Futhark_Hackers_Futhark,
author = {The Futhark Hackers},
title = {{Futhark}},
url = {https://github.com/diku-dk/futhark}
}
```
```bibtex
@inproceedings{henriksen2017futhark,
  title={Futhark: purely functional GPU-programming with nested parallelism and in-place array updates},
  author={Henriksen, Troels and Serup, Niels GW and Elsman, Martin and Henglein, Fritz and Oancea, Cosmin E},
  booktitle={Proceedings of the 38th ACM SIGPLAN Conference on Programming Language Design and Implementation},
  pages={556*571},
  year={2017}
}
```
```bibtex
@phdthesis{henriksen2017design,
  title={Design and implementation of the Futhark programming language},
  author={Henriksen, Troels},
  year={2017},
  school={University of Copenhagen, Faculty of Science [Department of Computer Science]}
}
```

## Standard ML host

The SML host uses MLKit and smlfut to call the existing Futhark engine through
its C API. It loads local, little-endian GGUF v2/v3 Qwen3 models containing
unquantized f16/f32 tensors and runs f32 or f16 inference. It supports
multi-turn Qwen3 chat, thinking mode, Futhark tool calls, and streaming output.
Commands are `quit`, `clear`, `think`, `nothink`, `show`, and `hide`.
The C, multicore, and CUDA backends can be selected at build time.

### Build and run

Install Futhark, MLKit with `Real16`, `Real32Array`, and `PackReal32Little`
support, smlfut with the MLKit target, smlpkg, a C compiler, and `make`.
The build fetches the pinned sml-unicode v0.2.0 dependency using `smlpkg sync`.
Python is not needed to build or run the SML host.

```sh
make SMLFUT=$HOME/gits/smlfut/smlfut
./build/chat-sml --model Qwen3-0.6B-f16.gguf --cs 2048
```

The defaults are `FUTHARK_BACKEND=c TYPE=f32`. To build for an NVIDIA GPU:

```sh
export CUDA_HOME=/usr/local/cuda
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
make FUTHARK_BACKEND=cuda TYPE=f16
./build/cuda-f16/chat-sml --model Qwen3-0.6B-f16.gguf --cs 1024
```

CUDA requires the NVIDIA driver and CUDA development toolkit, including NVRTC.
See the [Futhark CUDA documentation](https://futhark.readthedocs.io/en/latest/man/futhark-cuda.html).
Select `TYPE=f32` for full precision, or `FUTHARK_BACKEND=multicore` for parallel
CPU execution. In f16 mode, matrix weights and the KV cache use half precision;
normalization weights remain f32, matching `qwen-f16.fut`.

Each backend/precision has its own `build/BACKEND-TYPE/` executable, bindings,
and object caches. `build/chat-sml` points to the last selected executable.
Application SML sources are copied into each variant to isolate MLKit's object
cache; edit the originals in `sml/`, not the generated copies. CUDA kernel
caches are stored in `qwen-cuda-f32.cache` or `qwen-cuda-f16.cache` in the working
directory. These are disposable and are ignored by Git.

Use a model already on disk; the host does not download model weights.
For a single response with a generation limit:

```sh
./build/chat-sml --model Qwen3-0.6B-f16.gguf --prompt "Hello!" --max-new-tokens 64
```

`--cnt` controls the number of tokens per generation batch (default 5).
`--cs` sets the KV-cache capacity (default 1024 on CUDA, 8192 on CPU). Weights plus the f32 cache
require several GB of memory for Qwen3-0.6B; choose a smaller context when
needed. Full prompts are checked before inference, and generation stops at
EOS, the generation limit, or the context limit. `clear` resets conversation
and logical cache reuse without reloading weights.

### Thinking and tools

Thinking is off by default. Start with `--think`, or use `think` and `nothink`
during a conversation. Reasoning is captured separately from the answer and
hidden in normal output. Progress on stderr reports prompt processing and
hidden thinking/tool tokens, with updates at most once every five seconds
within a phase. The sequential C backend can take minutes to finish reasoning.
`show` displays the full prompts and generated output,
including reasoning and tool payloads; `hide` returns to answer-only output.
Reasoning is retained during tool follow-ups and removed from older assistant
messages when the next user turn is formatted, following the Qwen3 template.
The host recomputes the shared token prefix so the KV cache remains consistent.

Enable the available Futhark tools with `--tools`:

```sh
./build/chat-sml --model Qwen3-0.6B-f16.gguf --cs 2048 --tools
```

The build discovers entries named `tool_NAME_DESCRIPTION` with signature
`(n: i64): i64` from the Futhark manifest and generates typed smlfut dispatch.
The current tools are `sumN1` (sum of the first n integers) and `sumN2` (sum of
their squares). Completed calls run after EOS, their results are added as tool
messages, and generation resumes automatically. Multiple calls and successive
tool rounds are supported. No tool executes unless `--tools` is enabled.

Arguments must be integers from 0 through 1,000,000: these demo tools allocate
an `iota(n+1)` array. Unknown tools and invalid arguments produce JSON error
responses that the model can handle. Malformed or incomplete call blocks are
reported without execution. `--max-tool-rounds N` limits tool batches per user
turn (default 8); `--max-new-tokens` is a total generation budget across that
turn's assistant/tool follow-ups. Context capacity is checked for each prompt.

GGUF parsing, tensor transformations, byte-level BPE, and chat orchestration
are implemented in SML. NFC comes from sml-unicode's `unicode-normalise.mlb`;
a handwritten scanner implements the Qwen splitting rules using generated
Unicode 17 letter/number ranges. A rank heap handles BPE merges, and streaming
retains incomplete UTF-8 sequences between batches.

`sml/large-file.c` supplies a checked 64-bit seek while MLKit's installed POSIX
wrapper still uses C-int offsets. No parsing or tensor conversion is done in
C. The generated bindings and objects are kept in `build/`; `make clean-sml`
removes them. The downloaded `lib/` tree is ignored; `sml.pkg` pins its source.

### Host modules and ownership

`gguf.sig`, `weights.sig`, `engine.sig`, and `tokenizer.sig` define the public
interfaces. GGUF models/readers, engines, and tokenizers are abstract; engine
callers use `capacity` and `eos` rather than implementation records.

`Options` parses arguments into a typed configuration without I/O. `Main`
dispatches to `ChatApp` for interactive chat or `Diagnostics` for inspection,
reference-test protocols, and benchmarking. `Host` shares startup between the
two front ends. Existing command-line flags and chat commands are retained.

`Resource` releases owned handles in reverse acquisition order, attempts every
cleanup, and preserves an action's exception when cleanup also fails.
`Gguf.withReader` and `Engine.withModel` bound resource lifetimes. Weight loading
scopes temporary arrays; generation scopes input/output handles and transfers
a replacement cache to the engine after output extraction succeeds. Diagnostic
parameter projections are also scoped. `Engine.useContext` and `useParams`
provide borrowed handles for tools and inspection; callbacks must not free or
retain those handles. Failed inference closes the engine.

f16 matrix tensors are unpacked directly into `Real16Array` with
`PackReal16Little`, preserving their source bits and avoiding an f32 round trip. f32 source tensors use round-to-nearest
conversion when loaded into f16 matrices. Normalization weights stay f32, and
tensor reads fill a reusable 64 KiB `Word8Array` with `Posix.IO.readArr`.
Short reads continue into the same buffer, and GGUF strings use slice copies.
Metadata and tensor directories stay in vectors instead of being copied to
lists. A row wider than 64 KiB uses one additional row buffer.

`TensorLayout` expresses row-major, transpose, and attention reshaping as
contiguous destination runs. `Real16ArraySlice.modifyi` and
`Real32ArraySlice.modifyi` fill those runs, checking destination bounds per
slice and removing per-element scatter-index division. Decoders borrow the
input buffer only during each row-batch callback; no full raw-tensor copy is
needed. Input byte decoding still uses checked Basis packing operations.

Cached token-vector copies use `VectorSlice.vector`, and the common-prefix
scan uses `VectorSlice.findi`. Unicode scalar parsing initializes its result
vector directly after counting leading bytes. Token decoding copies into a
sized character buffer; UTF-8 cleanup copies spans into a lazy buffer only
when replacement is needed and returns valid input unchanged. BPE heap/link
updates and Unicode-category binary searches retain indexed access.

### Validation

The Python requirements are needed only for reference checks:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make check-sml SMLFUT=$HOME/gits/smlfut/smlfut
make check-parity PYTHON=.venv/bin/python
make check-conversation PYTHON=.venv/bin/python
make check-model PYTHON=.venv/bin/python MODEL=Qwen3-0.6B-f16.gguf
```

The same backend/precision selectors apply to checks, for example:

```sh
make FUTHARK_BACKEND=cuda TYPE=f16 check-model PYTHON=.venv/bin/python
```

`check-host` checks cleanup failure paths, engine lifetime, option validation,
all 65,536 half bit patterns (including signed zeros and NaNs), chunk-boundary
permutations, wide-row buffers, f32-to-f16 rounding, and preparation from f32
source matrices. It also compares every cell of row-major, transpose, and
attention layouts across partial batches and head boundaries. A separate
text driver checks 4,690 malformed UTF-8, valid-prefix streaming, and scalar
offset cases against Python's codecs.
Both `check-parity` and `check-model` include it.

`check-parity` creates a tiny deterministic two-layer GGUF with shuffled tensor
ordering. It compares weight shapes and sampled values, token IDs and decoded
text, pre-tokenizer splits, basic chat templates, fixed-token generation,
cache reuse and invalidation, EOS, and bounded contexts against the Python
host and a separately compiled Futhark server. It also exercises interactive
multi-turn chat, show/hide, clear, and oversized-prompt rejection. `check-model`
runs the same checks using the local model; it takes longer and uses more RAM.

`check-conversation` is a fast, standalone reasoning/tool check. Scripted model
token batches run through the production session loop, including real Futhark
tool dispatch. The checks cover reasoning capture and hiding, show mode,
multiple/successive calls, escaped JSON, invalid requests, disabled tools,
budgets, context limits, and prefix invalidation when reasoning is removed.
Both `check-parity` and `check-model` include these checks; full-model mode also
compares the prompts against the GGUF's actual Jinja template.

The reference host now wraps the splitting expression in `tokenizers.Regex`:
passing a plain string treated the pattern as a literal. Both hosts therefore
use the intended Qwen splitting rules. The SML NFC implementation uses Unicode
17; older Rust normalization tables in Python tokenizers may differ for newly
introduced characters. The tests check the Kirat Rai normalization explicitly
against its Unicode 17 result.

### Comparing host performance

The benchmark driver needs the Python test requirements. Run one GPU workload
at a time and retain the raw results outside the checkout:

```sh
.venv/bin/python tests/benchmark.py --exe build/cuda-f16/chat-sml \
  --backend cuda --type f16 --model Qwen3-0.6B-f16.gguf \
  --context 1024 --tokens 32 --runs 3 --out /tmp/fuchat-cuda-f16
```

Both hosts receive identical token IDs, clear logical cache state between
repetitions, perform one warmup, and decode one token per entry call. Generated
IDs must match. Prefill timing includes the first generated token; generation
throughput measures subsequent tokens. The report contains median rates,
individual timings, generated IDs/text, and startup/loading times. Loading
includes different host-specific initialization and diagnostic work, so those
times should be interpreted separately from inference throughput.

GPU memory is sampled with `nvidia-smi` every 100 ms, including loading and
warmup. The recorded peak is approximate total device usage, including desktop
processes; baseline usage is also recorded. This is not an allocator-level peak
measurement. Smaller contexts help fit the model on GPUs with limited VRAM.

### Initial ThinkPad CUDA measurements

Measured on 2026-11-01 with the NVIDIA RTX A500 Laptop GPU (4 GiB), CUDA 12.8,
Futhark 0.28.2, MLKit 4.7.26, and smlfut 1.6.5. Qwen3-0.6B-f16.gguf was used
for both precisions, with a 1024-token context, a 20-token non-thinking prompt,
32 generated tokens, one warmup, and three measured repetitions. Both hosts
produced identical IDs within each precision; f32 and f16 need not agree.

| Precision | Host | Prompt tokens/s | Decode tokens/s | Peak GPU MiB |
| --- | --- | ---: | ---: | ---: |
| f32 | MLKit | 14.44 | 1.16 | 3771 |
| f32 | Python | 10.14 | 0.60 | 3771 |
| f16 | MLKit | 58.65 | 10.01 | 2527 |
| f16 | Python | 58.49 | 10.01 | 2527 |

Rates are medians using the methodology above. Peak memory includes an 11 MiB
desktop baseline. These results describe this configuration, not tuned GPU
kernel performance. f16 provides the practical configuration for this 4 GiB
GPU; its warm inference rates are effectively equal between hosts.

SML loading took about 40 seconds for f32 and 53 seconds for f16; Python took
about 11 and 6 seconds respectively. Those initialization paths do different
work. These are the initial port measurements, before direct f16 loading was added;
the follow-up measurements below cover the updated loader.

### Direct f16 loader measurements

| Host / loader | Engine loading (s) | Prompt tokens/s | Decode tokens/s | Peak GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| SML, initial loader | 52.60 | 58.65 | 10.01 | 2527 |
| SML, direct f16 loader | 49.24 | 79.05 | 13.19 | 2527 |
| Python, reference | 5.97 | 57.16 | 9.96 | 2527 |

The direct loader measured 49.24 seconds versus the initial 52.60 seconds;
this is a modest change and the startup gap with Python remains substantial.
Loading measures context, weight preparation, and cache initialization, with
metadata/tokenizer work outside that interval. Throughput varied between runs;
the Futhark kernels were unchanged. All generated IDs matched Python. These
use the same context/prompt/output settings and measurement method as above.
Final raw measurements are retained on the ThinkPad in
`~/fuchat-test-tools/results/host-refactor/bench-cuda-f16-direct/`.

### Bulk slice and buffer measurements

| Host / loader | Engine loading (s) | Prompt tokens/s | Decode tokens/s | Peak GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| SML, before bulk operations | 49.24 | 79.05 | 13.19 | 2527 |
| SML, bulk slice/buffer loader | 22.43 | 58.42 | 10.01 | 2527 |
| Python, reference | 5.93 | 58.10 | 10.02 | 2527 |

Bulk operations reduced measured engine loading from 49.24 to 22.43 seconds
(about 2.2 times faster). Warm generation remains approximately 10 tokens/s,
matching the Python host, and all generated IDs match. The context, prompt,
output length, warmup, and repetition counts follow the methodology above.
The Futhark kernels are unchanged; throughput varied between laptop runs.

A separate two-token startup probe completed in 21.87 seconds, including
20.31 seconds of engine loading, with a maximum host resident set of about
556 MiB (`/usr/bin/time -v`). The normal input buffer is 64 KiB; tensor output
arrays are unchanged and no additional full-tensor copy is needed. Raw
measurements and check logs are retained on the ThinkPad under
`~/fuchat-test-tools/results/bulk-host/`.

### Unicode category data

Only regeneration requires Python. Download Unicode 17.0.0 `UnicodeData.txt`
from <https://www.unicode.org/Public/17.0.0/ucd/>, then run:

```sh
python3 tools/generate-categories.py /path/to/UnicodeData.txt > sml/UnicodeCategories.sml
```

The generator checks a pinned source hash. The Unicode data license is
included in `sml/unicode-license.txt` alongside the generated table.
