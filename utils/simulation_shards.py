"""
Sharded storage of simulation datasets.

A dataset produced by `simulating_neurons/submit_simulate_neuron_and_create_dataset.py` is one
folder per simulation:

    <split>/simulation_<n>/exc_weighted_spikes.npz     scipy CSR, (segments x ms), weights 1..4
    <split>/simulation_<n>/inh_weighted_spikes.npz     scipy CSR, (segments x ms), weights 1..3
    <split>/simulation_<n>/voltage.h5                  'somatic_voltage', float64, (ms,)
    <split>/simulation_<n>/summary.pkl                 output spike times, firing rates, args
    <split>/summary.pkl                                dataset level summary (read by the trainer)

That is ~5 files and ~2-8 MB per simulation, i.e. 1.5 M files and 500 GB for the 24 models
of the paper. This module packs the same information into a few HDF5 files per split:

    <split>/shard_000.h5, shard_001.h5, ...            SIMS_PER_SHARD simulations each
    <split>/summary.pkl                                unchanged

with, inside every shard, one group per simulation:

    sim_<n>/exc_indptr             uint32  (segments+1,)   CSR row pointer
    sim_<n>/exc_indices_delta      uint16  (nnz,)          column index, delta coded within a row
    sim_<n>/exc_data               uint8   (nnz,)          synaptic weight (spike count) 1..255
    sim_<n>/inh_indptr, inh_indices_delta, inh_data        the same for inhibition
    sim_<n>/somatic_voltage        float32 (ms,)
    sim_<n>/output_spike_times     int64   (spikes,)
    sim_<n>.attrs                  the scalar fields of simulation_<n>/summary.pkl, and the
                                   simulation args as json

Delta coding the column indices (the first index of a row is stored as is, every later one as
the distance from its predecessor) plus HDF5's byte shuffle and gzip makes a shard 2.4-3x
smaller than the npz+h5 files it replaces, and the whole thing is lossless except for the
somatic voltage, which is stored as float32 (~1e-6 mV).

`SimulationData` in training_nets/train_neuron_tcn.py reads either layout, so a downloaded
sharded dataset can be trained on directly.
"""

import os
import json
import glob
import pickle
import argparse

import h5py
import numpy as np
from scipy import sparse

FORMAT_VERSION = 1
SIMS_PER_SHARD = 500
SHARD_GLOB = 'shard_*.h5'
SHARD_NAME = 'shard_{:03d}.h5'
COMPRESSION = dict(compression='gzip', compression_opts=6, shuffle=True)

# scalar fields of a per simulation summary.pkl that go into the group attributes. Everything
# else in that pickle is either an array we store as a dataset, or args (stored as json).
SUMMARY_SCALAR_KEYS = [
    'len_exc_netcons', 'len_inh_netcons',
    'input_count_exc_spikes', 'input_average_exc_spikes_per_second',
    'input_count_exc_spikes_per_super_synapse', 'input_average_exc_spikes_per_super_synapse_per_second',
    'input_count_weighted_exc_spikes', 'input_average_weighted_exc_spikes_per_second',
    'input_count_weighted_exc_spikes_per_super_synapse', 'input_average_weighted_exc_spikes_per_super_synapse_per_second',
    'input_count_inh_spikes', 'input_average_inh_spikes_per_second',
    'input_count_inh_spikes_per_super_synapse', 'input_average_inh_spikes_per_super_synapse_per_second',
    'input_count_weighted_inh_spikes', 'input_average_weighted_inh_spikes_per_second',
    'input_count_weighted_inh_spikes_per_super_synapse', 'input_average_weighted_inh_spikes_per_super_synapse_per_second',
    'average_exc_initial_neuron_weight', 'average_inh_initial_neuron_weight',
    'output_firing_rate', 'output_firing_rate_after_initialization',
    'simulation_duration_in_ms',
    'average_somatic_voltage', 'average_clipped_somatic_voltage',
]
SUMMARY_ARRAY_KEYS = ['output_spike_times', 'output_isi', 'output_spike_times_after_initialization',
                      'output_isi_after_initialization']


# ----------------------------------------------------------------------------------------------
# encoding / decoding of one CSR matrix
# ----------------------------------------------------------------------------------------------

def encode_csr(matrix):
    """
    scipy CSR (segments x ms) -> (indptr uint32, indices_delta uint16, data uint8).

    The delta of the first nonzero of every row is its absolute column, the delta of every later
    nonzero is the distance from the previous nonzero in that row.
    """
    matrix = sparse.csr_matrix(matrix)
    matrix.sort_indices()
    indices = matrix.indices.astype(np.int64)
    indptr = matrix.indptr.astype(np.int64)
    data = matrix.data

    if matrix.shape[1] > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"{matrix.shape[1]} columns do not fit uint16 indices")
    if data.size and (data.min() < 0 or data.max() > np.iinfo(np.uint8).max or np.any(data != np.round(data))):
        raise ValueError("weights must be small non-negative integers to be stored as uint8")

    delta = indices.copy()
    if indices.size > 1:
        delta[1:] = np.diff(indices)
    row_starts = indptr[:-1][indptr[:-1] < indices.size]
    delta[row_starts] = indices[row_starts]
    return indptr.astype(np.uint32), delta.astype(np.uint16), data.astype(np.uint8)


def decode_csr(indptr, indices_delta, data, shape):
    """Inverse of encode_csr, returns a scipy CSR matrix with float64 data (as the npz files had)."""
    indptr = np.asarray(indptr, dtype=np.int64)
    delta = np.asarray(indices_delta, dtype=np.int64)
    if delta.size == 0:
        return sparse.csr_matrix(shape, dtype=np.float64)
    cumulative = np.cumsum(delta)
    counts = np.diff(indptr)
    # base of a row = cumulative sum just before the row starts (0 for the first nonzero)
    starts = indptr[:-1]
    base = np.where(starts > 0, cumulative[np.maximum(starts - 1, 0)], 0)
    indices = cumulative - np.repeat(base, counts)
    return sparse.csr_matrix((np.asarray(data, dtype=np.float64), indices.astype(np.int32), indptr.astype(np.int32)),
                             shape=shape)


# ----------------------------------------------------------------------------------------------
# reading / writing simulations
# ----------------------------------------------------------------------------------------------

def _to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_simulation_folder(simulation_folder):
    """Read one simulation_<n> folder into a dict (the unit that write_shard stores)."""
    exc = sparse.load_npz(os.path.join(simulation_folder, 'exc_weighted_spikes.npz'))
    inh = sparse.load_npz(os.path.join(simulation_folder, 'inh_weighted_spikes.npz'))
    with h5py.File(os.path.join(simulation_folder, 'voltage.h5'), 'r') as h:
        voltage = h['somatic_voltage'][:]
    summary = pickle.load(open(os.path.join(simulation_folder, 'summary.pkl'), 'rb'))
    return dict(exc=exc, inh=inh, somatic_voltage=voltage, summary=summary)


def write_simulation_group(group, simulation):
    """Store one simulation (as returned by load_simulation_folder) into an h5py group."""
    for tag in ('exc', 'inh'):
        matrix = simulation[tag]
        indptr, delta, data = encode_csr(matrix)
        group.attrs[f'{tag}_shape'] = np.asarray(matrix.shape, dtype=np.int64)
        group.create_dataset(f'{tag}_indptr', data=indptr, chunks=(len(indptr),), **COMPRESSION)
        if len(delta):
            group.create_dataset(f'{tag}_indices_delta', data=delta, chunks=(len(delta),), **COMPRESSION)
            group.create_dataset(f'{tag}_data', data=data, chunks=(len(data),), **COMPRESSION)
        else:
            group.create_dataset(f'{tag}_indices_delta', data=delta)
            group.create_dataset(f'{tag}_data', data=data)

    voltage = np.asarray(simulation['somatic_voltage'], dtype=np.float32)
    group.create_dataset('somatic_voltage', data=voltage, chunks=(len(voltage),), **COMPRESSION)

    summary = simulation['summary']
    for key in SUMMARY_ARRAY_KEYS:
        if key in summary:
            group.create_dataset(key, data=np.asarray(summary[key], dtype=np.int64))
    for key in SUMMARY_SCALAR_KEYS:
        if key in summary:
            group.attrs[key] = _to_native(summary[key])
    if 'args' in summary:
        args = summary['args']
        args = vars(args) if isinstance(args, argparse.Namespace) else args
        group.attrs['args_json'] = json.dumps(args, default=str)


def read_simulation_group(group, what=('exc', 'inh', 'somatic_voltage', 'output_spike_times')):
    """
    Read a simulation back from its h5py group.

    Returns a dict with (a subset of) 'exc', 'inh' (dense float64 arrays, as `.A` of the npz
    matrices gave), 'somatic_voltage' (float64), 'output_spike_times' (int64), and 'summary'
    (the scalar attrs + args as a Namespace) when asked for.
    """
    out = {}
    for tag in ('exc', 'inh'):
        if tag in what:
            shape = tuple(int(x) for x in group.attrs[f'{tag}_shape'])
            out[tag] = decode_csr(group[f'{tag}_indptr'][:], group[f'{tag}_indices_delta'][:],
                                  group[f'{tag}_data'][:], shape).toarray()
    if 'somatic_voltage' in what:
        out['somatic_voltage'] = group['somatic_voltage'][:].astype(np.float64)
    if 'output_spike_times' in what:
        out['output_spike_times'] = group['output_spike_times'][:]
    if 'summary' in what:
        summary = {k: _to_native(v) for k, v in group.attrs.items() if k not in ('args_json', 'exc_shape', 'inh_shape')}
        for key in SUMMARY_ARRAY_KEYS:
            if key in group:
                summary[key] = group[key][:]
        if 'args_json' in group.attrs:
            summary['args'] = argparse.Namespace(**json.loads(group.attrs['args_json']))
        out['summary'] = summary
    return out


def write_shard(shard_path, simulation_folders, simulation_indices, attrs=None):
    """
    Write one shard from a list of simulation folders.

    The shard is written under a temporary name unique to this process and renamed into place
    when complete, so a reader never sees a partial shard and two writers racing on the same
    shard (e.g. a job submitted twice) cannot corrupt each other - the last rename wins.
    """
    tmp_path = f'{shard_path}.{os.getpid()}.tmp'
    with h5py.File(tmp_path, 'w') as h:
        h.attrs['format_version'] = FORMAT_VERSION
        h.attrs['simulation_indices'] = np.asarray(simulation_indices, dtype=np.int64)
        for key, value in (attrs or {}).items():
            h.attrs[key] = value
        for index, folder in zip(simulation_indices, simulation_folders):
            write_simulation_group(h.create_group(f'sim_{index}'), load_simulation_folder(folder))
    os.replace(tmp_path, shard_path)


# ----------------------------------------------------------------------------------------------
# locating simulations inside a sharded split
# ----------------------------------------------------------------------------------------------

def is_sharded(split_folder):
    return len(glob.glob(os.path.join(split_folder, SHARD_GLOB))) > 0


class ShardIndex:
    """
    Maps simulation index -> (shard file, group name) for a sharded split folder, and hands out
    per-process open h5py files (safe to use from DataLoader workers: files are opened lazily in
    whichever process asks first, and never shared across a fork).
    """

    def __init__(self, split_folder):
        self.split_folder = split_folder
        self.shard_paths = sorted(glob.glob(os.path.join(split_folder, SHARD_GLOB)))
        if not self.shard_paths:
            raise FileNotFoundError(f"no {SHARD_GLOB} in {split_folder}")
        self.simulation_to_shard = {}
        for path in self.shard_paths:
            with h5py.File(path, 'r') as h:
                for index in h.attrs['simulation_indices']:
                    self.simulation_to_shard[int(index)] = path
        self._open_files = {}
        self._pid = None

    def __len__(self):
        return len(self.simulation_to_shard)

    @property
    def simulation_indices(self):
        return sorted(self.simulation_to_shard)

    def _file(self, path):
        pid = os.getpid()
        if self._pid != pid:
            self._open_files = {}
            self._pid = pid
        if path not in self._open_files:
            self._open_files[path] = h5py.File(path, 'r')
        return self._open_files[path]

    def group(self, simulation_index):
        path = self.simulation_to_shard[int(simulation_index)]
        return self._file(path)[f'sim_{int(simulation_index)}']

    def read(self, simulation_index, what=('exc', 'inh', 'somatic_voltage', 'output_spike_times')):
        return read_simulation_group(self.group(simulation_index), what)

    def close(self):
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files = {}
