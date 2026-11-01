#!/usr/bin/env python3
"""Check buffered UTF-8 replacement/streaming and scalar offsets independently."""
import argparse
import codecs
from pathlib import Path
import random
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--exe', required=True)
args = p.parse_args()
samples = [b'', bytes(range(128)), bytes(range(256))]
samples += [bytes([i]) for i in range(256)]
for text in ['é', 'α', '🙂', '\U0010ffff', 'ASCII\x00é🙂end']:
    data = text.encode()
    samples += [data[:i] for i in range(len(data)+1)]
samples += [b'valid'+data+b'tail' for data in
            [b'\x80', b'\xc0\xaf', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xe2\x82', b'\xff']]
rng = random.Random(931)
samples += [rng.randbytes(rng.randrange(32)) for _ in range(2048)]
cases = [(data, final) for data in samples for final in [False, True]]
with tempfile.TemporaryDirectory(prefix='fuchat-text-') as directory:
    fixture = Path(directory)/'utf8.tsv'
    fixture.write_text(''.join(f'{int(final)}\t{data.hex()}\n' for data,final in cases))
    result = subprocess.run([args.exe, str(fixture)], text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == len(cases)
    for (data,final),line in zip(cases, lines):
        clean,tail,scalars = line.split('\t')
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        expected = decoder.decode(data, final=final)
        # Buffer only prefixes that can still become valid UTF-8. CPython's
        # incremental decoder defers some already-invalid truncated sequences
        # (e.g. ED A9); strict decoding identifies those without duplicating
        # the SML byte-validation algorithm.
        pending = decoder.getstate()[0]
        if pending:
            try:
                pending.decode('utf-8')
            except UnicodeDecodeError as error:
                if error.reason != 'unexpected end of data':
                    expected += decoder.decode(b'', final=True)
        assert bytes.fromhex(clean).decode() == expected, (data,final,clean,expected)
        assert bytes.fromhex(tail) == decoder.getstate()[0], (data,final,tail)
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            assert scalars == 'invalid', (data,scalars)
        else:
            offset = 0
            expected = []
            for char in text:
                expected.append(f'{ord(char)}:{offset}')
                offset += len(char.encode())
            assert scalars == ','.join(expected), (data,scalars,expected)
    print(f'{len(cases)} buffered UTF-8/streaming/scalar cases match Python')
