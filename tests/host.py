#!/usr/bin/env python3
"""Check scoped ownership and direct half loading, including f32 source matrices."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from gguf import GGUFWriter, GGUFReader
import parity

p = argparse.ArgumentParser()
p.add_argument('--exe', required=True)
p.add_argument('--host-exe', required=True)
p.add_argument('--type', choices=['f16','f32'], required=True)
args = p.parse_args()
with tempfile.TemporaryDirectory(prefix='fuchat-host-') as directory:
    work = Path(directory)
    model = work / 'tiny.gguf'
    parity.tiny_model(model, matrix_dtype=np.float32)
    broken = work / 'truncated.gguf'
    broken.write_bytes(model.read_bytes()[:-400])
    edge = work / 'edge.gguf'
    values = np.array([0., -0., 1., -1., 1+2**-11, 1+3*2**-11, 2**-25,
                       3*2**-25, 2**-14, 65504., 65520., -65520., np.inf, -np.inf], dtype=np.float32)
    writer = GGUFWriter(edge, 'qwen3')
    writer.add_tensor('half_bits', np.arange(65536, dtype=np.uint16).view(np.float16))
    writer.add_tensor('float_values', values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    expected = work / 'expected.bin'
    with np.errstate(over='ignore'):
        expected.write_bytes(values.astype('<f2').tobytes())
    subprocess.run([str(Path(args.host_exe).resolve()), str(model), str(broken), str(edge), str(expected)], check=True)
    parity.check_weights(Path(args.exe).resolve(), model, GGUFReader(model), args.type)
    print('f32 source matrix preparation matches NumPy')
