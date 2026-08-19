"""
Download the paper's simulations and trained networks from Hugging Face.

    # the test simulations of one model (2500 simulations, 1-5 GB) and its three trained networks
    python data_release/download.py --model Rat_L5b_PC_2_Hay_passive_dends_simple_soma --splits test

    # train + test of one model (10000 + 2500 simulations, 4-23 GB)
    python data_release/download.py --model Rat_L5b_PC_2_Hay_passive_dends_simple_soma --splits train test

    # everything (~200 GB)
    python data_release/download.py --all

Files land under --output_folder (default: ./paper_data) as

    paper_data/simulations/models/<model>/{train,test}/shard_*.h5, summary.pkl
    paper_data/simulations/models/<model>/dataset_summary.pkl
    paper_data/tcns/models/<model>/d3_w128_t{0,1,2}/...

so that

    python training_nets/train_neuron_tcn.py --simulation_dataset_folder paper_data/simulations/models/<model> ...
    python calculate_fci.py --neuron_tcn_folder paper_data/tcns/models/<model>/d3_w128_t0

work directly. Needs `pip install huggingface_hub`.
"""

import os
import sys
import argparse

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_FOLDER)

from data_release import models as M  # noqa: E402

DATASET_REPO = os.environ.get('FCI_DATASET_REPO', 'i-do-ai/fci-neuron-simulations')
TCNS_REPO = os.environ.get('FCI_TCNS_REPO', 'i-do-ai/fci-neuron-tcns')


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', nargs='*', default=[], choices=M.release_names(), metavar='MODEL',
                        help='release name(s), i.e. folder names under simulating_neurons/neuron_models/')
    parser.add_argument('--all', action='store_true', help='all 24 models')
    parser.add_argument('--splits', nargs='*', default=['test'], choices=['train', 'test'])
    parser.add_argument('--no_simulations', action='store_true')
    parser.add_argument('--no_tcns', action='store_true')
    parser.add_argument('--output_folder', default=os.path.join(os.getcwd(), 'paper_data'))
    parser.add_argument('--dataset_repo', default=DATASET_REPO)
    parser.add_argument('--tcns_repo', default=TCNS_REPO)
    return parser.parse_args()


def main():
    args = get_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    models = M.release_names() if args.all else args.model
    if not models:
        raise SystemExit("give --model <name> [<name> ...] or --all")

    if not args.no_simulations:
        patterns = ['README.md', 'models.json']
        for model in models:
            patterns.append(f'models/{model}/dataset_summary.pkl')
            for split in args.splits:
                patterns.append(f'models/{model}/{split}/*')
        folder = snapshot_download(repo_id=args.dataset_repo, repo_type='dataset', allow_patterns=patterns,
                                   local_dir=os.path.join(args.output_folder, 'simulations'))
        print(f"simulations: {folder}")

    if not args.no_tcns:
        patterns = ['README.md'] + [f'models/{model}/*' for model in models]
        folder = snapshot_download(repo_id=args.tcns_repo, repo_type='model', allow_patterns=patterns,
                                   local_dir=os.path.join(args.output_folder, 'tcns'))
        print(f"trained networks: {folder}")


if __name__ == '__main__':
    main()
