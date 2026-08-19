"""
Pack one split of a simulation dataset into HDF5 shards (see utils/simulation_shards.py).

    python data_release/pack_dataset.py \
        --simulation_dataset_folder /path/to/dataset \
        --split train \
        --output_folder /path/to/packed/dataset \
        --workers 4

reads /path/to/dataset/train/simulation_<n>/ and writes

    /path/to/packed/dataset/train/shard_000.h5, shard_001.h5, ...
    /path/to/packed/dataset/train/summary.pkl          (copied)
    /path/to/packed/dataset/dataset_summary.pkl        (/path/to/dataset/summary.pkl, copied)

Shards that already exist are skipped, so a killed run can simply be restarted. Every shard
is written to a temporary name (unique per process) and renamed when complete, so a half
written shard is never mistaken for a finished one, and two runs racing on the same split
cannot corrupt each other.

The packed split is what `SimulationData` (training_nets/train_neuron_tcn.py) reads when it
finds shard_*.h5 in its folder, so `--simulation_dataset_folder /path/to/packed/dataset` works
for training exactly like the original folder did.
"""

import os
import re
import sys
import time
import glob
import shutil
import logging
import argparse
import multiprocessing

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_FOLDER)

from utils.simulation_shards import write_shard, SIMS_PER_SHARD, SHARD_NAME  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--simulation_dataset_folder', required=True)
    parser.add_argument('--split', required=True, help='train / valid / test')
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--sims_per_shard', type=int, default=SIMS_PER_SHARD)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--model_name', default=None, help='stored in every shard (default: dataset folder name)')
    parser.add_argument('--only_shards', type=int, nargs='*', default=None, help='pack only these shard numbers')
    return parser.parse_args()


def list_simulations(split_folder):
    """Sorted (index, folder) of every complete simulation_<n> in a split folder."""
    sims = []
    for folder in glob.glob(os.path.join(split_folder, 'simulation_*')):
        m = re.fullmatch(r'simulation_(\d+)', os.path.basename(folder))
        if not m:
            continue
        needed = ('exc_weighted_spikes.npz', 'inh_weighted_spikes.npz', 'voltage.h5', 'summary.pkl')
        if all(os.path.exists(os.path.join(folder, f)) for f in needed):
            sims.append((int(m.group(1)), folder))
        else:
            logger.warning(f"{folder} is incomplete, skipping")
    return sorted(sims)


def _pack_one(job):
    shard_path, folders, indices, attrs = job
    if os.path.exists(shard_path):
        return shard_path, 'exists', 0
    t = time.time()
    write_shard(shard_path, folders, indices, attrs=attrs)
    return shard_path, 'written', time.time() - t


def main():
    args = get_args()
    split_folder = os.path.join(args.simulation_dataset_folder, args.split)
    out_split = os.path.join(args.output_folder, args.split)
    os.makedirs(out_split, exist_ok=True)
    model_name = args.model_name or os.path.basename(os.path.normpath(args.simulation_dataset_folder))

    sims = list_simulations(split_folder)
    logger.info(f"{split_folder}: {len(sims)} simulations -> {out_split}")
    if not sims:
        raise SystemExit("nothing to pack")

    # the summaries the trainer reads, copied as they are
    for src, dst in ((os.path.join(split_folder, 'summary.pkl'), os.path.join(out_split, 'summary.pkl')),
                     (os.path.join(args.simulation_dataset_folder, 'summary.pkl'), os.path.join(args.output_folder, 'dataset_summary.pkl'))):
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)

    jobs = []
    for shard_number, start in enumerate(range(0, len(sims), args.sims_per_shard)):
        if args.only_shards is not None and shard_number not in args.only_shards:
            continue
        chunk = sims[start:start + args.sims_per_shard]
        indices = [i for i, _ in chunk]
        folders = [f for _, f in chunk]
        attrs = dict(model=model_name, split=args.split, shard=shard_number, sims_per_shard=args.sims_per_shard)
        jobs.append((os.path.join(out_split, SHARD_NAME.format(shard_number)), folders, indices, attrs))

    t0 = time.time()
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            for path, status, seconds in pool.imap_unordered(_pack_one, jobs):
                logger.info(f"{status:8s} {path} ({seconds:.0f}s)")
    else:
        for job in jobs:
            path, status, seconds = _pack_one(job)
            logger.info(f"{status:8s} {path} ({seconds:.0f}s)")
    total = sum(os.path.getsize(j[0]) for j in jobs)
    logger.info(f"done: {len(jobs)} shards, {total/1e9:.2f} GB, {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
