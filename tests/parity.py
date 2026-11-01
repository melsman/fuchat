#!/usr/bin/env python3
"""Independent reference checks for the SML host. Uses chat.py's real pipeline.

Run with the project's Python requirements installed. Without --model, builds
an unquantized two-layer fixture with shuffled tensors and a small byte BPE.
--model checks the locally available Qwen model, without downloading weights.
"""
import argparse
import contextlib
import io
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFReader, GGUFWriter
from jinja2 import Environment, BaseLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chat


def alphabet():
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in keep}
    for i, b in enumerate(b for b in range(256) if b not in mapping):
        mapping[b] = chr(256 + i)
    return [mapping[b] for b in range(256)]


def tiny_model(path, eos=257):
    vocab = alphabet()
    types = [1] * 256
    for token in ['<|im_start|>', '<|im_end|>', '<think>', '</think>', '<tool_call>', '</tool_call>']:
        vocab.append(token)
        types.append(3)
    rules = ['h e', 'he l', 'hel l', 'hell o', 'Ġ w', 'Ġw o', 'Ġwo r', 'Ġwor l', 'Ġworl d', 'Ċ Ċ']
    for rule in rules:
        vocab.append(''.join(rule.split()))
        types.append(1)
    w = GGUFWriter(path, 'qwen3')
    for k, v in {'block_count': 2, 'embedding_length': 8, 'feed_forward_length': 12,
                 'attention.head_count': 2, 'attention.head_count_kv': 1,
                 'attention.key_length': 128}.items():
        w.add_uint32('qwen3.' + k, v)
    w.add_string('tokenizer.ggml.model', 'gpt2')
    w.add_string('tokenizer.ggml.pre', 'qwen2')
    w.add_token_list(vocab)
    w.add_token_types(types)
    w.add_token_merges(rules)
    w.add_uint32('tokenizer.ggml.eos_token_id', eos)
    # Reference for the basic (no reasoning/tool calls) rendering subset.
    w.add_chat_template("{%- for m in messages %}{{- '<|im_start|>' + m.role + '\n' }}"
                        "{%- if m.role == 'assistant' and loop.last %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}"
                        "{{- m.content + '<|im_end|>\n' }}{%- endfor %}"
                        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{%- endif %}")
    rng = np.random.default_rng(42)
    shapes = {'token_embd.weight': (len(vocab), 8), 'output.weight': (len(vocab), 8),
              'output_norm.weight': (8,)}
    for b in [1, 0]:  # GGUF order must not determine block order.
        for name, shape in {'attn_norm': (8,), 'ffn_norm': (8,), 'ffn_gate': (12, 8),
                            'ffn_up': (12, 8), 'ffn_down': (8, 12), 'attn_q': (256, 8),
                            'attn_k': (128, 8), 'attn_v': (128, 8), 'attn_output': (8, 256),
                            'attn_q_norm': (128,), 'attn_k_norm': (128,)}.items():
            shapes[f'blk.{b}.{name}.weight'] = shape
    for name, shape in shapes.items():
        data = rng.uniform(0.5, 1.5, shape).astype(np.float32) if len(shape) == 1 else rng.normal(0, 0.1, shape).astype(np.float16)
        w.add_tensor(name, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def sml(exe, model, *args, input=None, success=True):
    p = subprocess.run([str(exe), '--model', str(model), *map(str, args)], input=input,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (p.returncode == 0) != success:
        raise AssertionError(f'SML exit {p.returncode}: {p.stderr}')
    return p


def check_weights(exe, model, reader):
    lines = sml(exe, model, '--weights', '--cs', 64).stdout.splitlines()
    assert len(lines) == 14
    for line in lines:
        name, shape, values = line.split('\t')
        if name in ['token_embd', 'output', 'output_norm']:
            t = chat.get_gguf_tensor(reader, name + '.weight')
            if name == 'output':
                t = t.T
        else:
            t = chat.collect_gguf_tensor(reader, name + '.weight', reader.get_field('qwen3.block_count').contents())
        if name in ['attn_q', 'attn_k', 'attn_v']:
            t = t.reshape((*t.shape[:-1], -1, 128)).transpose(0, 2, 1, 3)
        assert tuple(map(int, shape.split(','))) == t.shape, name
        # Avoid a complete flattened/transposed copy of a large weight field.
        points = [0, min(1, t.size - 1), t.size // 3, t.size // 2, t.size - 1]
        expected = np.array([t[np.unravel_index(i, t.shape)] for i in points], dtype=np.float32)
        got = np.array([float(x.replace('~', '-')) for x in values.split(',')], dtype=np.float32)
        np.testing.assert_array_equal(got, expected, err_msg=name)
    print('14 weight shapes and sampled values match')


def check_tokenizer(exe, model, reader, work):
    tokenizer = chat.make_tokenizer(reader)
    samples = ['', 'hello world', "'ſword", "I'm WE'RE he'll I'd can't", 'a\u030a Å Å ① ﬁ',
               'αβ 中文 日本語 한국어 한글', '🙂👩\u200d💻𐐀', '١٢３④²',
               '\t  x\r\n\nnext  \t', '\x00\x01\x7f', 'a'*2048, ' hello!!!\r\n',
               '<|im_start|>user\nHi<|im_end|>\n', '<think>\n\n</think>\n\n']
    rng = random.Random(710)
    chars = list("aZ 09!'\t\n\r_-") + ['ø', 'α', '中', '١', '²', '🙂', '\u0301', '\u00a0', '\u2003', '\u2028']
    samples += [''.join(rng.choice(chars) for _ in range(rng.randrange(0, 96))) for _ in range(500)]
    path = work / 'texts.hex'
    path.write_text(''.join(s.encode().hex() + '\n' for s in samples))
    output = sml(exe, model, '--tokenizer-test', path).stdout.splitlines()
    assert len(output) == len(samples)
    for text, line in zip(samples, output):
        ids, decoded, pieces = line.split('\t')
        got = list(map(int, ids.split(','))) if ids else []
        expected = tokenizer.encode(text).ids
        assert got == expected, (repr(text), got, expected)
        assert bytes.fromhex(decoded).decode() == tokenizer.decode(expected, skip_special_tokens=False), repr(text)
        # Offsets in the Python API are Unicode scalar positions.
        normalized = tokenizer.normalizer.normalize_str(text)
        expected_pieces = [normalized[a:b] for _, (a, b) in tokenizer.pre_tokenizer.pre_tokenize_str(normalized)]
        got_pieces = [bytes.fromhex(x).decode() for x in pieces.split(',')] if pieces else []
        assert got_pieces == expected_pieces, (repr(text), got_pieces, expected_pieces)
    # The installed Rust NFC tables predate Kirat Rai. Check the Unicode 17
    # result explicitly rather than copying that reference-version limitation.
    path.write_text('\U00016d63\U00016d68'.encode().hex() + '\n')
    line = sml(exe, model, '--tokenizer-test', path).stdout.strip().split('\t')
    expected = tokenizer.encode('\U00016d6a').ids
    assert list(map(int, line[0].split(','))) == expected
    print(f'{len(samples)} tokenizer/decoder/pre-tokenizer cases match; Unicode 17 NFC regression passed')
    return tokenizer


def check_render(exe, model, reader, work):
    template = Environment(loader=BaseLoader).from_string(reader.get_field('tokenizer.chat_template').contents())
    histories = [[{'role': 'user', 'content': 'Hello'}],
                 [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'αβ?'}],
                 [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi!'}],
                 [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi!'},
                  {'role': 'user', 'content': 'And now?'}]]
    cases = [(h, gen) for h in histories for gen in [False, True]]
    path = work / 'render.tsv'
    path.write_text(''.join(('1' if gen else '0') + '\t' + '\t'.join(
        m['role'] + ':' + m['content'].encode().hex() for m in h) + '\n' for h, gen in cases))
    output = sml(exe, model, '--render-test', path).stdout.splitlines()
    for (h, gen), actual in zip(cases, output):
        expected = template.render(messages=h, tools=False, add_generation_prompt=gen, enable_thinking=False)
        assert bytes.fromhex(actual).decode() == expected, (h, gen, bytes.fromhex(actual).decode(), expected)
    assert len(output) == len(cases)
    print(f'{len(cases)} basic Qwen chat renderings match Jinja')


def check_inference(exe, model, reader, tokenizer, work):
    # Build an independent server with the same generated sequential C backend.
    server_path = work / 'qwen-server'
    subprocess.run(['futhark', 'c', '--server', 'qwen-f32.fut', '-o', str(server_path)], check=True)
    old = chat.futhark_server.Server
    chat.futhark_server.Server = lambda *args: old(str(server_path))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            llm = chat.LLM(reader, 'f32', 64, 0)
    finally:
        chat.futhark_server.Server = old
    steps, expected = [], []
    def generate(ids, count):
        reused = int(llm.shared_prefix_length(ids, llm.tokens))
        out = llm.gen(np.array(ids, dtype=np.int64), count)
        new = out[len(ids):].tolist()
        steps.append(str(count) + '\t' + ','.join(map(str, ids)))
        expected.append((reused, new))
        return out.tolist()
    try:
        base = tokenizer.encode('Hello').ids
        # Prefill, multi-token continuation, prompt extension, prefix drift,
        # clear, and the last slot in a bounded context.
        out = generate(base, 1)
        out = generate(out, 2)
        generate(out + tokenizer.encode(' world').ids, 1)
        generate(tokenizer.encode('Different').ids, 2)
        steps.append('clear')
        llm.kvcached = 0
        llm.tokens = np.array([], dtype=np.int64)
        generate(base, 1)
        near = [base[0]] * 64
        generate(near, 1)
        steps.append('clear')
        llm.tokens = np.array([], dtype=np.int64)
        template = Environment(loader=BaseLoader).from_string(reader.get_field('tokenizer.chat_template').contents())
        history, answers, prompts = [], [], []
        for message in ['hello', 'again']:
            history.append({'role': 'user', 'content': message})
            rendered = template.render(messages=history, tools=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(rendered)
            ids = tokenizer.encode(rendered).ids
            out = generate(ids, 1)
            if out[-1] != llm.eos_token_id:
                out = generate(out, 2)
            visible = out[len(ids):]
            if visible and visible[-1] == llm.eos_token_id:
                visible = visible[:-1]
            answer = tokenizer.decode(visible, skip_special_tokens=False)
            answers.append(answer)
            history.append({'role': 'assistant', 'content': answer.strip()})
    finally:
        llm.__exit__(None, None, None)
    trace = work / 'trace.tsv'
    trace.write_text('\n'.join(steps) + '\n')
    output = sml(exe, model, '--trace', trace, '--cs', 64).stdout.splitlines()
    actual = [(int(line.split('\t')[0]), list(map(int, line.split('\t')[1].split(','))) if line.split('\t')[1] else []) for line in output]
    assert actual == expected, (actual, expected)
    print(f'{len(expected)} fixed-token/cache/context traces match the Python/Futhark server')
    return answers, prompts


def check_eos(exe, work):
    model = work / 'eos-base.gguf'
    tiny_model(model)
    reader = GGUFReader(model)
    tokenizer = chat.make_tokenizer(reader)
    old = chat.futhark_server.Server
    chat.futhark_server.Server = lambda *args: old(str(work / 'qwen-server'))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            llm = chat.LLM(reader, 'f32', 64, 0)
    finally:
        chat.futhark_server.Server = old
    try:
        ids = np.array(tokenizer.encode('Hello').ids, dtype=np.int64)
        predicted = int(llm.gen(ids, 1)[-1])
        assert predicted > 0  # The Futhark entry uses zero as its loop sentinel.
        llm.tokens = np.array([], dtype=np.int64)
        llm.eos_token_id = predicted
        reference = llm.gen(ids, 5)[len(ids):].tolist()
        assert reference == [predicted]
    finally:
        llm.__exit__(None, None, None)
    eos_model = work / 'eos.gguf'
    tiny_model(eos_model, eos=predicted)
    trace = work / 'eos.tsv'
    trace.write_text('5\t' + ','.join(map(str, ids)) + '\n')
    line = sml(exe, eos_model, '--cs', 64, '--trace', trace).stdout.strip().split('\t')
    assert line == ['0', str(predicted)], line
    print('EOS stops generation after one token, matching the reference')


def check_chat(exe, model, answers, prompts):
    p = sml(exe, model, '--cs', 128, '--max-new-tokens', 3,
            input='show\nhello\nhide\nagain\nclear\nhello\nquit\n')
    expected = ('Qwen3 / MLKit. Commands: quit, clear, think, nothink, show, hide.\nchat> \nchat> '
                + prompts[0] + '\n' + answers[0] + '\nchat> chat> ' + answers[1]
                + '\nchat> Context cleared\nchat> ' + answers[0] + '\nchat> ')
    assert p.stdout == expected, (p.stdout, expected)
    assert p.stderr.count('generated tokens') == 3, p.stderr
    assert p.stderr.count('Reused 0 cached tokens') >= 2, p.stderr
    # A too-long prompt is rejected without invoking Futhark out of bounds.
    p = sml(exe, model, '--cs', 4, '--prompt', 'hello', '--max-new-tokens', 1)
    assert 'Context is full' in p.stderr
    print('interactive multi-turn, show/hide, clear, and context rejection passed')


def check_bad_files(exe, model, work):
    for name, data in [('bad-magic', b'NOPE' + bytes(32)), ('truncated', b'GGUF\x03\0\0\0'),
                       ('bad-version', b'GGUF\xff\0\0\0' + bytes(32))]:
        path = work / (name + '.gguf')
        path.write_bytes(data)
        sml(exe, path, '--weights', success=False)
    # Regression for MLKit's C-int POSIX seek truncating offsets above 1 GiB:
    # a sparse GGUF data region is exercised by the real-model weight checks.
    print('malformed/truncated GGUF files are rejected')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exe', default='./build/chat-sml')
    p.add_argument('--model')
    args = p.parse_args()
    exe = Path(args.exe).resolve()
    with tempfile.TemporaryDirectory(prefix='fuchat-parity-') as d:
        work = Path(d)
        model = Path(args.model).resolve() if args.model else work / 'tiny.gguf'
        if not args.model:
            tiny_model(model)
        reader = GGUFReader(model)
        check_bad_files(exe, model, work)
        check_weights(exe, model, reader)
        tokenizer = check_tokenizer(exe, model, reader, work)
        check_render(exe, model, reader, work)
        answers, prompts = check_inference(exe, model, reader, tokenizer, work)
        check_eos(exe, work)
        check_chat(exe, model, answers, prompts)
    print('All host parity checks passed')

if __name__ == '__main__':
    main()
