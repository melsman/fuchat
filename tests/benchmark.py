#!/usr/bin/env python3
"""Compare warm SML/Python host timings for one backend and precision.

Both hosts prefill the same token IDs and decode one token per call, clearing
logical cache state between repetitions. Prefill includes the first generated
token. Startup/loading is reported separately; GPU memory is sampled at 100ms.
Raw measurements remain in --out (or a newly created temporary directory).
"""
import argparse
import contextlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

import numpy as np
from gguf import GGUFReader
from jinja2 import Environment, BaseLoader
import parity


def gpu_used():
    if not shutil.which('nvidia-smi'):
        return None
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                       text=True, capture_output=True, check=True)
    return int(p.stdout.splitlines()[0])


@contextlib.contextmanager
def monitor(path, metrics):
    baseline = gpu_used()
    process = None
    with path.open('w') as output:
        if baseline is not None:
            process = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits', '--loop-ms=100'], stdout=output, stderr=subprocess.DEVNULL)
        try:
            yield
        finally:
            if process:
                process.terminate()
                process.wait()
    if baseline is not None:
        samples = [int(s) for s in path.read_text().splitlines() if s.strip().isdigit()]
        metrics.update(gpu_baseline_mib=baseline, gpu_peak_mib=max(samples + [baseline]), sample_ms=100)


def summarize(result):
    runs = result['runs']
    result['median_prompt_tps'] = statistics.median(r['prompt_tokens']/r['prompt_seconds'] for r in runs)
    result['median_generation_tps'] = statistics.median((r['generated_tokens']-1)/r['generation_seconds'] for r in runs)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exe', required=True)
    p.add_argument('--model', default='Qwen3-0.6B-f16.gguf')
    p.add_argument('--backend', default='cuda', choices=['c','cuda','multicore'])
    p.add_argument('--type', default='f32', choices=['f32','f16'])
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--prompt', default='Write a story about a hedgehog.')
    p.add_argument('--out')
    args = p.parse_args()
    if args.tokens < 2 or args.runs < 1:
        p.error('need at least two generated tokens and one repetition')
    out = Path(args.out).resolve() if args.out else Path(tempfile.mkdtemp(prefix='fuchat-benchmark-'))
    out.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(args.model)
    tokenizer = parity.chat.make_tokenizer(reader)
    template = Environment(loader=BaseLoader).from_string(reader.get_field('tokenizer.chat_template').contents())
    prompt = template.render(messages=[dict(role='user', content=args.prompt)], tools=False,
                             add_generation_prompt=True, enable_thinking=False)
    ids = np.array(tokenizer.encode(prompt).ids, dtype=np.int64)
    if len(ids) + args.tokens > args.context:
        p.error('context must hold prompt plus generated tokens')
    ids_file = out / 'ids.csv'
    ids_file.write_text(','.join(map(str, ids)))
    server = out / 'qwen-server'
    with (out / 'server-build.log').open('w') as log:
        subprocess.run(['futhark', args.backend, '--server', f'qwen-{args.type}.fut', '-o', str(server)],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    gpu = subprocess.run(['nvidia-smi'], text=True, capture_output=True).stdout if shutil.which('nvidia-smi') else ''
    (out / 'gpu.txt').write_text(gpu)
    sml_memory = {}
    with monitor(out / 'sml-memory.csv', sml_memory), (out / 'sml-stderr.log').open('w') as log:
        result = subprocess.run([str(Path(args.exe).resolve()), '--model', args.model, '--cs', str(args.context),
            '--benchmark', str(ids_file), '--benchmark-tokens', str(args.tokens), '--benchmark-runs', str(args.runs)],
            text=True, capture_output=False, stdout=subprocess.PIPE, stderr=log, check=True)
    sml = summarize(json.loads(result.stdout))
    assert (sml['backend'], sml['precision'], sml['context']) == (args.backend, args.type, args.context), \
        'executable does not match requested backend, precision, or context'
    assert len(sml['runs']) == args.runs
    (out / 'sml.json').write_text(result.stdout)
    old = parity.chat.futhark_server.Server
    # Warm just the reference program's JIT cache before timing model loading.
    cache = out / 'python.cache'
    with old(str(server), '--cache=' + str(cache)) as warm:
        warm.cmd('entry_points')
    parity.chat.futhark_server.Server = lambda *unused: old(str(server), '--cache=' + str(cache))
    py_memory = {}
    with monitor(out / 'python-memory.csv', py_memory), (out / 'python-stderr.log').open('w') as log:
        try:
            start = time.perf_counter()
            with contextlib.redirect_stdout(log):
                llm = parity.chat.LLM(reader, args.type, args.context, 0)
            load_seconds = time.perf_counter()-start
        finally:
            parity.chat.futhark_server.Server = old
        def one():
            llm.tokens = np.array([], dtype=np.int64)
            llm.kvcached = 0
            start = time.perf_counter()
            tokens = llm.gen(ids, 1)
            prefill = time.perf_counter()-start
            start = time.perf_counter()
            while len(tokens)-len(ids) < args.tokens and len(tokens) <= args.context and tokens[-1] != llm.eos_token_id:
                tokens = llm.gen(tokens, 1)
            decode = time.perf_counter()-start
            return dict(prompt_seconds=prefill, generation_seconds=decode, prompt_tokens=len(ids),
                        generated_tokens=len(tokens)-len(ids), ids=tokens[len(ids):].tolist())
        try:
            one()
            runs = [one() for _ in range(args.runs)]
        finally:
            llm.__exit__(None, None, None)
    python = summarize(dict(load_seconds=load_seconds, warmups=1, runs=runs))
    (out / 'python.json').write_text(json.dumps(python, indent=2))
    for a, b in zip(sml['runs'], python['runs']):
        assert a['ids'] == b['ids'], (a['ids'], b['ids'])
    sml.update(sml_memory)
    python.update(py_memory)
    report = dict(backend=args.backend, precision=args.type, context=args.context, prompt=args.prompt,
                  prompt_tokens=len(ids), requested_tokens=args.tokens, warmups=1, repetitions=args.runs,
                  ids_match=True, generated_text=tokenizer.decode(sml["runs"][0]["ids"], skip_special_tokens=False),
                  sml=sml, python=python, archive=str(out))
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
