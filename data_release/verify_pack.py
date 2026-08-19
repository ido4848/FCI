"""
Check a packed dataset split against the simulation folders it was packed from.

    python data_release/verify_pack.py \
        --simulation_dataset_folder /path/to/dataset --split test \
        --packed_folder /path/to/packed/dataset [--sample 50] [--all]

For every shard: every simulation of the original split is present exactly once. For a random
sample of simulations (or all of them with --all): the dense excitatory and inhibitory input
matrices are bit-identical to `sparse.load_npz(...).A`, the somatic voltage is within float32
precision, the output spike times and the scalar summary fields are equal.

Exits non-zero on the first mismatch, prints a one line summary otherwise.
"""

import os
import re
import sys
import glob
import time
import pickle
import argparse

import h5py
import numpy as np

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_FOLDER)

from utils.simulation_shards import ShardIndex, load_simulation_folder, SUMMARY_SCALAR_KEYS  # noqa: E402

VOLTAGE_TOLERANCE_MV = 1e-4


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--simulation_dataset_folder', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--packed_folder', required=True)
    parser.add_argument('--sample', type=int, default=50)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def original_indices(split_folder):
    out = []
    for folder in glob.glob(os.path.join(split_folder, 'simulation_*')):
        m = re.fullmatch(r'simulation_(\d+)', os.path.basename(folder))
        if m and os.path.exists(os.path.join(folder, 'summary.pkl')):
            out.append(int(m.group(1)))
    return sorted(out)


def compare(index, split_folder, shard_index):
    folder = os.path.join(split_folder, f'simulation_{index}')
    orig = load_simulation_folder(folder)
    got = shard_index.read(index, what=('exc', 'inh', 'somatic_voltage', 'output_spike_times', 'summary'))
    problems = []
    if not np.array_equal(got['exc'], orig['exc'].toarray()):
        problems.append('exc differs')
    if not np.array_equal(got['inh'], orig['inh'].toarray()):
        problems.append('inh differs')
    v_err = np.abs(got['somatic_voltage'] - orig['somatic_voltage']).max()
    if v_err > VOLTAGE_TOLERANCE_MV:
        problems.append(f'voltage differs by {v_err}')
    if not np.array_equal(got['output_spike_times'], orig['summary']['output_spike_times']):
        problems.append('output_spike_times differ')
    for key in SUMMARY_SCALAR_KEYS:
        if key in orig['summary'] and not np.isclose(got['summary'][key], orig['summary'][key], rtol=1e-12, atol=0):
            problems.append(f'summary[{key}] {got["summary"][key]} != {orig["summary"][key]}')
    return problems, v_err


def main():
    args = get_args()
    split_folder = os.path.join(args.simulation_dataset_folder, args.split)
    packed_split = os.path.join(args.packed_folder, args.split)
    t0 = time.time()

    expected = original_indices(split_folder)
    shard_index = ShardIndex(packed_split)
    got = shard_index.simulation_indices
    if got != expected:
        missing = sorted(set(expected) - set(got))[:10]
        extra = sorted(set(got) - set(expected))[:10]
        raise SystemExit(f"FAIL {packed_split}: {len(got)} packed vs {len(expected)} original simulations; missing {missing} extra {extra}")

    for name in ('summary.pkl',):
        a = os.path.join(split_folder, name)
        b = os.path.join(packed_split, name)
        if os.path.exists(a) and (not os.path.exists(b) or open(a, 'rb').read() != open(b, 'rb').read()):
            raise SystemExit(f"FAIL {b} missing or differs from {a}")

    if args.all:
        sample = expected
    else:
        rng = np.random.default_rng(args.seed)
        sample = sorted(rng.choice(expected, size=min(args.sample, len(expected)), replace=False).tolist())
        # always include the first and last simulation of every shard
        for path in shard_index.shard_paths:
            with h5py.File(path, 'r') as h:
                idx = h.attrs['simulation_indices']
                sample += [int(idx[0]), int(idx[-1])]
        sample = sorted(set(sample))

    max_v_err = 0
    for index in sample:
        problems, v_err = compare(index, split_folder, shard_index)
        max_v_err = max(max_v_err, v_err)
        if problems:
            raise SystemExit(f"FAIL {packed_split} simulation_{index}: {problems}")

    size = sum(os.path.getsize(p) for p in shard_index.shard_paths)
    print(f"OK {packed_split}: {len(expected)} simulations in {len(shard_index.shard_paths)} shards ({size/1e9:.2f} GB), "
          f"{len(sample)} compared, max voltage error {max_v_err:.2e} mV, {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
