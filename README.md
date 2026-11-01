# fuchat

## Introduction

Qwen3 LLM inference using the Futhark language + demo chat application

The inference engine has support for KV Cache & Prompt extension.

The chat app has support for
 - user/assistant chat
 - Thinking mode on/off
 - Tool calling of simple futhark entry points

The default model is Qwen3-0.6B and currently needs around 6GB of VRAM


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
unquantized f16/f32 tensors and currently runs f32 inference. It supports
ordinary multi-turn chat, a fixed Qwen3 non-thinking chat format, streaming,
and `quit`, `clear`, `show`, and `hide` commands. Thinking/tool-call handling
and GPU/f16 validation are later port milestones.

### Build and run

Install Futhark, MLKit with `Real16`, `Real32Array`, and `PackReal32Little`
support, smlfut with the MLKit target, smlpkg, a C compiler, and `make`.
The build fetches the pinned sml-unicode v0.2.0 dependency using `smlpkg sync`.
Python is not needed to build or run the SML host.

```sh
make SMLFUT=$HOME/gits/smlfut/smlfut
./build/chat-sml --model Qwen3-0.6B-f16.gguf --cs 2048
```

Use a model already on disk; the host does not download model weights.
For a single response with a generation limit:

```sh
./build/chat-sml --model Qwen3-0.6B-f16.gguf --prompt "Hello!" --max-new-tokens 64
```

`--cnt` controls the number of tokens per generation batch (default 5).
`--cs` sets the KV-cache capacity (default 8192). Weights plus the f32 cache
require several GB of memory for Qwen3-0.6B; choose a smaller context when
needed. Full prompts are checked before inference, and generation stops at
EOS, the generation limit, or the context limit. `clear` resets conversation
and logical cache reuse without reloading weights.

GGUF parsing, tensor transformations, byte-level BPE, and chat orchestration
are implemented in SML. NFC comes from sml-unicode's `unicode-normalise.mlb`;
a handwritten scanner implements the Qwen splitting rules using generated
Unicode 17 letter/number ranges. A rank heap handles BPE merges, and streaming
retains incomplete UTF-8 sequences between batches.

`sml/large-file.c` supplies a checked 64-bit seek while MLKit's installed POSIX
wrapper still uses C-int offsets. No parsing or tensor conversion is done in
C. The generated bindings and objects are kept in `build/`; `make clean-sml`
removes them. The downloaded `lib/` tree is ignored; `sml.pkg` pins its source.

### Validation

The Python requirements are needed only for reference checks:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make check-sml SMLFUT=$HOME/gits/smlfut/smlfut
make check-parity PYTHON=.venv/bin/python
make check-model PYTHON=.venv/bin/python MODEL=Qwen3-0.6B-f16.gguf
```

`check-parity` creates a tiny deterministic two-layer GGUF with shuffled tensor
ordering. It compares weight shapes and sampled values, token IDs and decoded
text, pre-tokenizer splits, basic chat templates, fixed-token generation,
cache reuse and invalidation, EOS, and bounded contexts against the Python
host and a separately compiled Futhark server. It also exercises interactive
multi-turn chat, show/hide, clear, and oversized-prompt rejection. `check-model`
runs the same checks using the local model; it takes longer and uses more RAM.

The reference host now wraps the splitting expression in `tokenizers.Regex`:
passing a plain string treated the pattern as a literal. Both hosts therefore
use the intended Qwen splitting rules. The SML NFC implementation uses Unicode
17; older Rust normalization tables in Python tokenizers may differ for newly
introduced characters. The tests check the Kirat Rai normalization explicitly
against its Unicode 17 result.

### Unicode category data

Only regeneration requires Python. Download Unicode 17.0.0 `UnicodeData.txt`
from <https://www.unicode.org/Public/17.0.0/ucd/>, then run:

```sh
python3 tools/generate-categories.py /path/to/UnicodeData.txt > sml/UnicodeCategories.sml
```

The generator checks a pinned source hash. The Unicode data license is
included in `sml/unicode-license.txt` alongside the generated table.
